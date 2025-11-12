// gain_MPV_fit_robust.groovy
// Per-wire ADC calibration with ROBUST background subtraction:
//   SUB = SIG - α*BKG - β   (α,β from robust LSQ in control window)
// Landau MPV per wire -> gain = MPV_ref / MPV_wire
// MPV_corr = MPV_wire * gain, errors from width/sqrt(N)
// Optional α linear-in-ADC (disabled by default)
// Deuteron purity via W^2, Δφ≈180°, optional banana, WF gate, EXACT_WIRE

import java.util.*
import javax.swing.JFrame
import java.io.File

import org.jlab.io.hipo.HipoDataSource
import org.jlab.io.base.DataEvent
import org.jlab.io.base.DataBank

import org.jlab.groot.data.*
import org.jlab.groot.graphics.EmbeddedCanvasTabbed
import org.jlab.groot.math.F1D
import org.jlab.groot.fitter.DataFitter

import org.jlab.jnp.utils.options.OptionStore
import org.jlab.jnp.utils.benchmark.ProgressPrintout
import groovy.transform.Field

// ---------------- Config ----------------
@Field double EBEAM=2.24d, M_D=1.875612d
@Field double W2_MIN=3.46d, W2_MAX=3.67d, DPHI_HALF=10.0d
@Field boolean BANANA_ON=false
@Field double BAN_SCALE=1000, BAN_C0=6.0, BAN_C1=-35.0, BAN_HALF=1.5

@Field int KF_NHITS_MIN=8; @Field double KF_CHI2_MAX=30
@Field double VZ_MIN=-20, VZ_MAX=+10; @Field boolean FD_ONLY=true
@Field double PT_SLICE_MIN=0.15, PT_SLICE_MAX=0.45
@Field long   MAXEV=-1L

@Field int ADC_NBINS=220; @Field double ADC_MAX=4500.0
@Field int PT_NBINS_2D=100; @Field double PT_MIN_2D=0.20, PT_MAX_2D=0.45
@Field int ADC_NBINS_2D=220; @Field double ADC_MIN_2D=0.0, ADC_MAX_2D=4500.0

@Field int DRAW_COLS=5, DRAW_ROWS=3; int PAGE_SIZE(){ DRAW_COLS*DRAW_ROWS }
enum AssocMode { EXACT_WIRE, LAYER_ONLY }
@Field AssocMode assocMode = AssocMode.EXACT_WIRE

@Field boolean PROTON_VETO_ON=false
@Field double  PROT_VETO_MIN=100, PROT_VETO_MAX=650

// Control window and robust settings
@Field double CTRL_LO=80, CTRL_HI=480
@Field boolean USE_LINEAR_ALPHA=false // if true: α(ADC)=α0+α1*(ADC-αx0)
@Field double ALPHA_X0=300.0
@Field double HUBER_C=1.345          // robust weight scale
@Field int    IRLS_MAX=12
@Field double IRLS_TOL=1e-4
@Field double PEDESTAL_EXCLUDE_MAX=60 // ignore bins below this in control fit

// Fit windows for Landau
@Field double FIT_LO=450.0, FIT_HI=2000.0

@Field Integer FORCE_TRACK_ID=null
@Field boolean DRAW_GREEN_FITS=true
@Field boolean DO_PER_WIRE_FITS=true

// ---------------- Helpers ----------------
static double deg0to360(double a){ double x=a%360; (x<0)? x+360 : x }
static double phiDeg(float px,float py){ deg0to360(Math.toDegrees(Math.atan2((double)py,(double)px))) }
static double dphi0to360(double pe,double pk){ double x=(pe-pk)%360; (x<0)? x+360 : x }
static boolean between(double x,double lo,double hi){ x>=lo && x<=hi }
static int Lenc(int sl,int l){ 10*sl+l }
static double clampADC(int A,double max){ Math.min(max-1e-6,(double)A) }
static boolean inBand(double pt, long sumADC_trk, double c0,double c1,double half,double scale){
  double y=((double)sumADC_trk)/scale
  double c=c0+c1*(pt-0.26)
  Math.abs(y-c)<=half
}

class YSym{ double ymin,ymax; YSym(double a,double b){ymin=a;ymax=b} }
YSym symmetricY(H1F h, double pad=1.15){
  int n=h.getAxis().getNBins()
  double ymin=0,ymax=0
  for(int b=0;b<n;b++){
    double y=h.getBinContent(b)
    if(b==0){ymin=y;ymax=y}else{ if(y<ymin) ymin=y; if(y>ymax) ymax=y }
  }
  double a=Math.max(Math.abs(ymin),Math.abs(ymax)); a=(a<=0)?1.0:a*pad
  new YSym(-a,+a)
}
void setPadRanges(def c,int p,double xmin,double xmax,double ymin,double ymax){
  try{ c.getPad(p).setAxisRange(xmin,xmax,ymin,ymax) }catch(Throwable t){}
}

// ---------------- Data structs ----------------
class RecP{ int pid; float px,py,pz,vx,vy,vz,vt,beta,chi2pid; byte charge; short status }
class KFRec{ int idx=-1,nhits=-1,trackid=-1; float px,py,chi2; double phiDeg=Double.NaN }

final class SLW{ final int sl,l,w; SLW(int a,int b,int c){sl=a;l=b;w=c}
  int hashCode(){ ((sl*1315423911)^(l*2654435761))^w }
  boolean equals(Object o){ if(!(o instanceof SLW))return false; SLW x=(SLW)o; x.sl==sl&&x.l==l&&x.w==w }
}
final class SL{ final int sl,l; SL(int a,int b){sl=a;l=b}
  int hashCode(){ (sl*1315423911)^l }
  boolean equals(Object o){ if(!(o instanceof SL))return false; SL x=(SL)o; x.sl==sl&&x.l==l }
}
final class WireKey{
  final int s,Lraw,c; WireKey(int s,int L,int c){this.s=s;this.Lraw=L;this.c=c}
  int hashCode(){ (s*73856093)^(Lraw*19349663)^(c*83492791) }
  boolean equals(Object o){ if(!(o instanceof WireKey))return false; WireKey k=(WireKey)o; k.s==s&&k.Lraw==Lraw&&k.c==c }
  String toString(){ String.format("S%d L%02d C%d",s,Lraw,c) }
}
final class PairH{
  final H1F sig,bkg,sub
  PairH(String tag,String title,int nb,double lo,double hi){
    sig=new H1F("sig_"+tag,title+" (SIG);ADC;Counts",nb,lo,hi)
    bkg=new H1F("bkg_"+tag,title+" (BKG);ADC;Counts",nb,lo,hi)
    sub=new H1F("sub_"+tag,title+" (SUB=SIG-α·BKG-β);ADC;Counts",nb,lo,hi)
    sig.setLineColor(1); bkg.setLineColor(2); sub.setLineColor(4)
  }
}

// ---------------- Readers & association ----------------
RecP getElectronREC(DataEvent ev){
  if(!ev.hasBank("REC::Particle")) return null
  DataBank b=ev.getBank("REC::Particle")
  int best=-1
  for(int i=0;i<b.rows();i++){
    if(b.getInt("pid",i)!=11) continue
    float vz=b.getFloat("vz",i); short st=b.getShort("status",i)
    if(vz<VZ_MIN||vz>VZ_MAX) continue
    if(FD_ONLY && st>=0) continue
    best=i; break
  }
  if(best<0) return null
  RecP e=new RecP()
  e.pid=b.getInt("pid",best)
  e.px=b.getFloat("px",best); e.py=b.getFloat("py",best); e.pz=b.getFloat("pz",best)
  e.vx=b.getFloat("vx",best); e.vy=b.getFloat("vy",best); e.vz=b.getFloat("vz",best); e.vt=b.getFloat("vt",best)
  e.charge=b.getByte("charge",best); e.beta=b.getFloat("beta",best); e.chi2pid=b.getFloat("chi2pid",best); e.status=b.getShort("status",best)
  return e
}
double W_from_e(double Ebeam, RecP e){
  double Ee=Math.sqrt(e.px*e.px+e.py*e.py+e.pz*e.pz)
  double qx=-e.px, qy=-e.py, qz=Ebeam-e.pz, q0=Ebeam-Ee
  double Eh=M_D+q0
  double w2=Eh*Eh-(qx*qx+qy*qy+qz*qz)
  (w2>0)? Math.sqrt(w2): Double.NaN
}
KFRec bestKF_BackToBack(DataEvent ev,double phi_e){
  KFRec out=new KFRec(); if(!ev.hasBank("AHDC::kftrack")) return out
  DataBank k=ev.getBank("AHDC::kftrack")
  double bestAbs=1e9
  for(int i=0;i<k.rows();i++){
    int nh=k.getInt("n_hits",i); if(nh<KF_NHITS_MIN) continue
    float chi2=k.getFloat("chi2",i); if(!Float.isNaN(chi2)&&chi2>KF_CHI2_MAX) continue
    float px=k.getFloat("px",i),py=k.getFloat("py",i)
    double pk=phiDeg(px,py)
    double d=Math.abs(dphi0to360(phi_e,pk)-180.0)
    if(d<bestAbs){ bestAbs=d; out.idx=i; out.px=px; out.py=py; out.chi2=chi2; out.nhits=nh; out.phiDeg=pk; out.trackid=k.getInt("trackid",i) }
  }
  return out
}
KFRec bestKF_Heavy(DataEvent ev){
  KFRec out=new KFRec(); if(!ev.hasBank("AHDC::kftrack")) return out
  DataBank k=ev.getBank("AHDC::kftrack")
  double best=-1
  for(int i=0;i<k.rows();i++){
    int nh=k.getInt("n_hits",i); if(nh<KF_NHITS_MIN) continue
    float chi2=k.getFloat("chi2",i); if(!Float.isNaN(chi2)&&chi2>KF_CHI2_MAX) continue
    int sadc=0; try{sadc=k.getInt("sum_adc",i)}catch(Exception ignore){}
    double score=(nh>0)? ((double)sadc)/nh : -1
    if(score>best){ best=score; out.idx=i; out.nhits=nh; out.chi2=chi2; out.px=k.getFloat("px",i); out.py=k.getFloat("py",i); out.phiDeg=phiDeg(out.px,out.py); out.trackid=k.getInt("trackid",i) }
  }
  return out
}
class AssocSets{ Set<SLW> slw=new HashSet<>(); Set<SL> sl=new HashSet<>() }
AssocSets buildAssocSetsForTrackId(DataEvent ev,int wantedId){
  AssocSets as=new AssocSets()
  if(!ev.hasBank("AHDC::hits")) return as
  DataBank h=ev.getBank("AHDC::hits")
  for(int i=0;i<h.rows();i++){
    if(h.getInt("trackid",i)!=wantedId) continue
    int sl=(h.getByte("superlayer",i)&0xFF), l=(h.getByte("layer",i)&0xFF), w=h.getInt("wire",i)
    as.slw.add(new SLW(sl,l,w)); as.sl.add(new SL(sl,l))
  }
  return as
}

// --------------- WF gate ---------------
Set<String> wfExplicitGood(DataEvent ev){
  HashSet<String> good=new HashSet<>(); if(!ev.hasBank("AHDC::wf")) return good
  DataBank w=ev.getBank("AHDC::wf")
  for(int i=0;i<w.rows();i++){
    int flag; try{flag=w.getInt("flag",i)}catch(Exception e){continue}
    if(flag!=0 && flag!=1) continue
    int s,Lraw,c
    try{
      int sl=w.getInt("superlayer",i), l=w.getInt("layer",i)
      s=w.getInt("sector",i); c=w.getInt("component",i); Lraw=Lenc(sl,l)
    }catch(Exception e){
      try{ s=w.getInt("sector",i); c=w.getInt("component",i); Lraw=w.getInt("layer",i) }catch(Exception ee){continue}
    }
    good.add(s+"#"+Lraw+"#"+c)
  }
  return good
}
boolean wfPassForAdcRow(DataBank a,int i,Set<String> good){
  int s=a.getInt("sector",i), Lraw=a.getInt("layer",i), c=a.getInt("component",i)
  if(good.contains(s+"#"+Lraw+"#"+c)) return true
  int wft=Integer.MIN_VALUE; try{wft=a.getInt("wfType",i)}catch(Exception ignore){}
  if(wft!=Integer.MIN_VALUE && wft>2) return false
  Double tot=null; try{ tot=(double)a.getFloat("timeOverThreshold",i)}catch(Exception ignore){}
  if(tot!=null && (tot<250.0||tot>1200.0)) return false
  return true
}

// --------------- Hists -----------------
@Field Map<WireKey,PairH>  histMap=new LinkedHashMap<>()
@Field Map<WireKey,H1F>    subGainMap=new LinkedHashMap<>()
@Field Map<WireKey,Double> alphaMap=new LinkedHashMap<>()
@Field Map<WireKey,Double> betaMap =new LinkedHashMap<>() // NEW β

PairH getPair(WireKey k){
  PairH p=histMap.get(k)
  if(p==null){
    String tag=k.toString().replace(' ','_')
    p=new PairH(tag, "ADC — "+k.toString()+" (p_{T} slice)", ADC_NBINS, 0.0, ADC_MAX)
    histMap.put(k,p)
  }
  return p
}

@Field H1F SUM_SIG=new H1F("sum_sig","Per-wire ADC (SIG, all wires);ADC;Counts",ADC_NBINS,0,ADC_MAX)
@Field H1F SUM_BKG=new H1F("sum_bkg","Per-wire ADC (BKG, all wires);ADC;Counts",ADC_NBINS,0,ADC_MAX)
@Field H1F SUM_SUB=new H1F("sum_sub","Per-wire ADC (SUB, all wires);ADC;Counts",ADC_NBINS,0,ADC_MAX)
@Field H1F SUM_SUB_FIT=null
@Field H1F SUM_SIG_GAIN_1D=new H1F("sum_sig_gain","Per-wire ADC (SIG, gain-corr);ADC_{corr};Counts",ADC_NBINS,0,ADC_MAX)
void styleSum(){ SUM_SIG.setLineColor(1); SUM_BKG.setLineColor(2); SUM_SUB.setLineColor(4) }

// 2D maps
@Field H2F PTADC_BEFORE=new H2F("ptadc_before","ADC vs p_{T} — BEFORE; p_{T}; ADC",PT_NBINS_2D,PT_MIN_2D,PT_MAX_2D,ADC_NBINS_2D,ADC_MIN_2D,ADC_MAX_2D)
@Field H2F PTADC_SIG   =new H2F("ptadc_sig","ADC vs p_{T} — SIG; p_{T}; ADC",PT_NBINS_2D,PT_MIN_2D,PT_MAX_2D,ADC_NBINS_2D,ADC_MIN_2D,ADC_MAX_2D)
@Field H2F PTADC_BKG   =new H2F("ptadc_bkg","ADC vs p_{T} — BKG; p_{T}; ADC",PT_NBINS_2D,PT_MIN_2D,PT_MAX_2D,ADC_NBINS_2D,ADC_MIN_2D,ADC_MAX_2D)
@Field H2F PTADC_SIG_GAIN=new H2F("ptadc_sig_gain","ADC_{corr} vs p_{T} — SIG; p_{T}; ADC_{corr}",PT_NBINS_2D,PT_MIN_2D,PT_MAX_2D,ADC_NBINS_2D,ADC_MIN_2D,ADC_MAX_2D)

// --------------- Robust control fit: (α,β) (and optional α1) ---------------
class AlphaCtrlRes { double a0=0, a1=0, beta=0; int nUsed=0; double Ssum=0, Bsum=0 }
double huberW(double r,double c){ double ar=Math.abs(r); (ar<=c)? 1.0 : (c/ar) } // classic Huber

AlphaCtrlRes fitAlphaBeta(H1F sig,H1F bkg,double amin,double amax, boolean linearAlpha){
  def ax=sig.getAxis(); int n=ax.getNBins()
  double xmin=ax.min(), xmax=ax.max(), bw=(xmax-xmin)/n

  // collect control rows (exclude pedestal)
  ArrayList<double[]> rows=new ArrayList<>()
  double Ssum=0, Bsum=0
  for(int b=0;b<n;b++){
    double xL=xmin+b*bw, xC=xL+0.5*bw
    if(xC<PEDESTAL_EXCLUDE_MAX) continue
    if(xC<amin || xC>amax)      continue
    double S=sig.getBinContent(b), B=bkg.getBinContent(b)
    if(B<=0 && S<=0) continue
    rows.add(new double[]{xC, S, B})
    Ssum+=S; Bsum+=B
  }
  AlphaCtrlRes out=new AlphaCtrlRes(); out.Ssum=Ssum; out.Bsum=Bsum
  if(rows.size()<6){ out.a0=(Bsum>0? Ssum/Bsum:0); out.beta=0; out.nUsed=rows.size(); return out }

  // parameters: θ = [a0, beta] or [a0, a1, beta]
  double a0= (Bsum>0? Ssum/Bsum : 1.0), a1=0.0, beta=0.0
  for(int it=0; it<IRLS_MAX; it++){
    double wSum=0
    double m11=0,m12=0, m22=0, rhs1=0, rhs2=0
    double m11L=0,m12L=0,m13L=0, m22L=0,m23L=0,m33L=0, r1L=0,r2L=0,r3L=0

    for(double[] t: rows){
      double x=t[0], S=t[1], B=t[2]
      double alpha = linearAlpha? (a0 + a1*(x-ALPHA_X0)) : a0
      double pred  = alpha*B + beta
      double r     = S - pred
      double w     = huberW(r, HUBER_C)* (1.0/Math.max(1.0,S+B)) // robust + variance weight
      wSum+=w

      if(!linearAlpha){
        // [a0, beta]
        m11+=w*(B*B); m12+=w*B; m22+=w
        rhs1+=w*B*S;  rhs2+=w*S
      }else{
        double bx=B, bx1=B*(x-ALPHA_X0), c1=1.0
        m11L+=w*bx*bx;     m12L+=w*bx*bx1;   m13L+=w*bx*c1
        m22L+=w*bx1*bx1;   m23L+=w*bx1*c1;   m33L+=w*c1*c1
        r1L +=w*bx*S;      r2L +=w*bx1*S;    r3L +=w*c1*S
      }
    }

    // solve
    if(!linearAlpha){
      double det=m11*m22 - m12*m12
      if(Math.abs(det)<1e-12) break
      double na0=( rhs1*m22 - m12*rhs2)/det
      double nb =( m11*rhs2 - m12*rhs1)/det
      double da=Math.abs(na0-a0)+Math.abs(nb-beta)
      a0=na0; beta=nb
      if(da<IRLS_TOL) break
    }else{
      // 3x3 solve (symmetric)
      double[][] M=[[m11L,m12L,m13L],[m12L,m22L,m23L],[m13L,m23L,m33L]]
      double[]   R=[r1L,r2L,r3L]
      // Gaussian elimination (tiny system)
      for(int k=0;k<3;k++){
        double piv=M[k][k]; if(Math.abs(piv)<1e-12) { piv=1e-12; M[k][k]=piv }
        for(int j=k;j<3;j++) M[k][j]/=piv
        R[k]/=piv
        for(int i=0;i<3;i++) if(i!=k){
          double f=M[i][k]
          for(int j=k;j<3;j++) M[i][j]-=f*M[k][j]
          R[i]-=f*R[k]
        }
      }
      double na0=R[0], na1=R[1], nb=R[2]
      double da=Math.abs(na0-a0)+Math.abs(na1-a1)+Math.abs(nb-beta)
      a0=na0; a1=na1; beta=nb
      if(da<IRLS_TOL) break
    }
  }
  out.a0=a0; out.a1=(linearAlpha? a1:0.0); out.beta=beta; out.nUsed=rows.size()
  return out
}

// --------------- MPV tools ----------------
double estimateMPVError(H1F h,double mpv,double xmin,double xmax){
  if(h==null || Double.isNaN(mpv) || mpv<=0) return 0.0
  def ax=h.getAxis(); int n=ax.getNBins()
  double sumW=0,sumVar=0
  for(int b=0;b<n;b++){
    double x=ax.getBinCenter(b); if(x<xmin||x>xmax) continue
    double c=h.getBinContent(b); if(c<=0) continue
    sumW+=c; double dx=x-mpv; sumVar+=c*dx*dx
  }
  if(sumW<=1.0) return 0.0
  double sigma=Math.sqrt(sumVar/sumW)
  sigma/Math.sqrt(sumW)
}
class Fit1D{ boolean ok=false; double amp=0,mpv=0,sig=0,mpvErr=0,chi2=Double.NaN; int ndf=0; H1F curve=null }
Fit1D fitLandauRange(H1F h,double xmin,double xmax,String tag){
  Fit1D out=new Fit1D(); if(h==null || h.integral()<80) return out
  int bMax=h.getMaximumBin(); double mpv0=h.getAxis().getBinCenter(bMax)
  double amp0=Math.max(1.0,h.getBinContent(bMax)), sig0=Math.max(30.0,(xmax-xmin)/12.0)
  if(mpv0<xmin || mpv0>xmax) mpv0=0.5*(xmin+xmax)
  F1D f=new F1D("f_"+tag,"[A]*landau(x,[MPV],[SIG])",xmin,xmax)
  f.setParameter(0,amp0); f.setParameter(1,mpv0); f.setParameter(2,sig0)
  try{ DataFitter.fit(f,h,"Q") }catch(Throwable t){ return out }
  out.amp=f.getParameter(0); out.mpv=f.getParameter(1); out.sig=f.getParameter(2)
  out.mpvErr=estimateMPVError(h,out.mpv,xmin,xmax)

  // chi2/ndf
  def ax=h.getAxis(); int n=ax.getNBins()
  int used=0; double cs=0
  for(int b=0;b<n;b++){
    double x=ax.getBinCenter(b); if(x<xmin||x>xmax) continue
    double y=h.getBinContent(b); if(y<=0) continue
    double yf=f.evaluate(x); double err=Math.sqrt(y); if(err<=0) err=1.0
    double d=(y-yf)/err; cs+=d*d; used++
  }
  int ndf=used-f.getNPars(); if(ndf>0){ out.chi2=cs; out.ndf=ndf }
  out.ok=(out.mpv>0 && !Double.isNaN(out.mpv))
  // draw curve
  H1F c=new H1F("fit_"+tag,"",ADC_NBINS,h.getAxis().min(),h.getAxis().max())
  def ax2=c.getAxis(); int nb=ax2.getNBins()
  for(int i=0;i<nb;i++){ double x=ax2.getBinCenter(i); c.setBinContent(i,(x>=xmin&&x<=xmax)? f.evaluate(x):0) }
  c.setLineColor(3); out.curve=c
  return out
}
class FitG{ boolean ok=false; double mpv=0,mpvErr=0,amp=0,sig=0,chi2=Double.NaN; int ndf=0 }
FitG fitGlobal(H1F h,double xmin,double xmax){
  FitG out=new FitG(); if(h.integral()<200) return out
  int bMax=h.getMaximumBin(); double mpv0=h.getAxis().getBinCenter(bMax)
  double amp0=Math.max(1.0,h.getBinContent(bMax)), sig0=Math.max(50.0,(xmax-xmin)/10.0)
  F1D f=new F1D("f_g","[A]*landau(x,[MPV],[SIG])",xmin,xmax)
  f.setParameter(0,amp0); f.setParameter(1,mpv0); f.setParameter(2,sig0)
  try{ DataFitter.fit(f,h,"Q") }catch(Throwable t){ return out }
  out.amp=f.getParameter(0); out.mpv=f.getParameter(1); out.sig=f.getParameter(2)
  out.mpvErr=estimateMPVError(h,out.mpv,xmin,xmax)

  def ax=h.getAxis(); int n=ax.getNBins(); int used=0; double cs=0
  for(int b=0;b<n;b++){
    double x=ax.getBinCenter(b); if(x<xmin||x>xmax) continue
    double y=h.getBinContent(b); if(y<=0) continue
    double yf=f.evaluate(x); double err=Math.sqrt(y); if(err<=0) err=1.0
    double d=(y-yf)/err; cs+=d*d; used++
  }
  int ndf=used-f.getNPars(); if(ndf>0){ out.chi2=cs; out.ndf=ndf }
  out.ok=(out.mpv>0 && !Double.isNaN(out.mpv))
  return out
}

// --------------- CLI ---------------
OptionStore opt=new OptionStore("pd_robust")
opt.addCommand("process","Per-wire ADC with robust background subtraction (α,β)")
def po=opt.getOptionParser("process")
["-nevent","-beam","-w2min","-w2max","-dphiHalf","-banana","-mode_valid","-vzmin","-vzmax","-fdonly",
 "-ptmin","-ptmax","-ctrlLo","-ctrlHi","-protVeto","-protMin","-protMax","-trackid"].each{ po.addOption(it,"") }
opt.parse(args); if(opt.getCommand()!="process"){ System.err.println("usage ..."); System.exit(1) }

// glob
List<String> expandGlob(String pat){
  ArrayList<String> out=new ArrayList<>(); if(pat==null) return out
  File f=new File(pat)
  if(f.exists()&&f.isFile()){ out.add(f.getPath()); return out }
  if(pat.indexOf('*')>=0||pat.indexOf('?')>=0){
    File parent=f.getParentFile(); if(parent==null) parent=new File(".")
    String rx="\\Q"+f.getName().replace("?","\\E.\\Q").replace("*","\\E.*\\Q")+"\\E"
    def re=java.util.regex.Pattern.compile(rx)
    File[] list=parent.listFiles(); if(list!=null) for(File ff:list) if(ff.isFile()&&re.matcher(ff.getName()).matches()) out.add(ff.getPath())
  }
  return out
}

// inputs
ArrayList<String> raw=new ArrayList<>(); for(String s: po.getInputList()) raw.add(s)
ArrayList<String> files=new ArrayList<>()
for(String s: raw){ if(s==null) continue; if(s.toLowerCase().endsWith(".hipo")) files.add(s); else if(s.indexOf('*')>=0||s.indexOf('?')>=0) files.addAll(expandGlob(s)) }
LinkedHashSet<String> uniq=new LinkedHashSet<>(files); files.clear(); files.addAll(uniq); Iterator<String> itf=files.iterator(); while(itf.hasNext()){ String f=itf.next(); if(!(new File(f).isFile())) itf.remove() }
if(files.isEmpty()){ System.err.println("No .hipo inputs."); System.exit(1) }

// options
def gv={String k-> po.getOption(k)?.getValue()}
try{ String v=gv("-nevent"); if(v!=null) MAXEV=Long.parseLong(v.trim()) }catch(Exception ignore){}
try{ String v=gv("-beam");   if(v!=null) EBEAM=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ String v=gv("-w2min");  if(v!=null) W2_MIN=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ String v=gv("-w2max");  if(v!=null) W2_MAX=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ String v=gv("-dphiHalf"); if(v!=null) DPHI_HALF=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ String v=gv("-banana"); if(v!=null) BANANA_ON=Boolean.parseBoolean(v.trim()) }catch(Exception ignore){}
try{ String v=gv("-mode_valid"); if(v!=null) BANANA_ON=v.trim().equalsIgnoreCase("banana") }catch(Exception ignore){}
try{ String v=gv("-vzmin"); if(v!=null) VZ_MIN=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ String v=gv("-vzmax"); if(v!=null) VZ_MAX=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ String v=gv("-fdonly"); if(v!=null) FD_ONLY=Boolean.parseBoolean(v.trim()) }catch(Exception ignore){}
try{ String v=gv("-ptmin"); if(v!=null) PT_SLICE_MIN=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ String v=gv("-ptmax"); if(v!=null) PT_SLICE_MAX=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ String v=gv("-ctrlLo"); if(v!=null) CTRL_LO=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ String v=gv("-ctrlHi"); if(v!=null) CTRL_HI=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ String v=gv("-protVeto"); if(v!=null) PROTON_VETO_ON=Boolean.parseBoolean(v.trim()) }catch(Exception ignore){}
try{ String v=gv("-protMin"); if(v!=null) PROT_VETO_MIN=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ String v=gv("-protMax"); if(v!=null) PROT_VETO_MAX=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ String v=gv("-trackid"); if(v!=null) FORCE_TRACK_ID=Integer.valueOf(v.trim()) }catch(Exception ignore){}

// ---------------- Pass 1: fill SIG/BKG ----------------
styleSum()
ProgressPrintout prog=new ProgressPrintout()
long seen=0

for(String fn: files){
  HipoDataSource R=new HipoDataSource()
  try{ R.open(fn) }catch(Exception ex){ System.err.println("Open fail "+fn+" : "+ex); continue }
  while(R.hasEvent()){
    DataEvent ev; try{ ev=R.getNextEvent() }catch(Exception ex){ break }
    seen++

    RecP e=getElectronREC(ev); if(e==null){ prog.updateStatus(); if(MAXEV>0&&seen>=MAXEV) break; else continue }
    double pt=Math.hypot(e.px,e.py)
    double phi_e=phiDeg(e.px,e.py)
    KFRec kf_bb=bestKF_BackToBack(ev,phi_e), kf_heavy=bestKF_Heavy(ev)
    if(kf_bb.idx<0||kf_heavy.idx<0){ prog.updateStatus(); if(MAXEV>0&&seen>=MAXEV) break; else continue }

    int assocId=(FORCE_TRACK_ID!=null)? FORCE_TRACK_ID.intValue() : kf_heavy.trackid
    if(assocId<0){ prog.updateStatus(); if(MAXEV>0&&seen>=MAXEV) break; else continue }
    AssocSets AS=buildAssocSetsForTrackId(ev,assocId)

    double W=W_from_e(EBEAM,e); double W2=(Double.isNaN(W)? Double.NaN: W*W)
    boolean okW=(!Double.isNaN(W2)&&between(W2,W2_MIN,W2_MAX))
    boolean okDP=(Math.abs(dphi0to360(phi_e,kf_bb.phiDeg)-180.0)<=DPHI_HALF)

    long sumTRK=0
    if(ev.hasBank("AHDC::adc")){
      DataBank a0=ev.getBank("AHDC::adc")
      for(int i=0;i<a0.rows();i++){
        int A=0; try{A=a0.getInt("ADC",i)}catch(Exception ignore){}
        if(A<=0) continue
        int Lraw=a0.getInt("layer",i), c=a0.getInt("component",i)
        int sl=Lraw/10, l=Lraw%10
        if(AS.slw.contains(new SLW(sl,l,c))) sumTRK+=(long)A
      }
    }
    try{ DataBank k=ev.getBank("AHDC::kftrack"); int sadc=k.getInt("sum_adc",kf_heavy.idx); if(sadc>0) sumTRK=(long)sadc }catch(Exception ignore){}
    boolean okBan=(!BANANA_ON) || inBand(pt,sumTRK,BAN_C0,BAN_C1,BAN_HALF,BAN_SCALE)
    boolean ELASTIC=okW && okDP && okBan

    if(ev.hasBank("AHDC::adc")){
      DataBank a=ev.getBank("AHDC::adc")
      Set<String> wfGood=wfExplicitGood(ev)
      for(int i=0;i<a.rows();i++){
        int A=0; try{A=a.getInt("ADC",i)}catch(Exception ignore){}
        if(A<=0) continue
        if(!wfPassForAdcRow(a,i,wfGood)) continue
        int Lraw=a.getInt("layer",i), c=a.getInt("component",i), s=a.getInt("sector",i)
        int sl=Lraw/10, l=Lraw%10
        boolean trkPass=(assocMode==AssocMode.EXACT_WIRE)? AS.slw.contains(new SLW(sl,l,c)) : AS.sl.contains(new SL(sl,l))
        if(!trkPass) continue

        PTADC_BEFORE.fill(pt,(double)A)
        WireKey wk=new WireKey(s,Lraw,c)
        PairH ph=getPair(wk)
        double x=clampADC(A,ADC_MAX)
        if(ELASTIC){
          if(!(PROTON_VETO_ON && x>=PROT_VETO_MIN && x<=PROT_VETO_MAX)){
            ph.sig.fill(x); SUM_SIG.fill(x); PTADC_SIG.fill(pt,x)
          }
        }else{
          ph.bkg.fill(x); SUM_BKG.fill(x); PTADC_BKG.fill(pt,x)
        }
      }
    }

    prog.updateStatus()
    if(MAXEV>0&&seen>=MAXEV) break
  }
  try{ R.close() }catch(Exception ignore){}
}

// --------------- α,β per wire & global ----------------
def arSUM=fitAlphaBeta(SUM_SIG,SUM_BKG,CTRL_LO,CTRL_HI,false)
double A_glob=arSUM.a0, B_glob=arSUM.beta

for(Map.Entry<WireKey,PairH> e: histMap.entrySet()){
  WireKey wk=e.getKey(); PairH ph=e.getValue()
  def ar=fitAlphaBeta(ph.sig, ph.bkg, CTRL_LO, CTRL_HI, USE_LINEAR_ALPHA)
  double a=ar.a0, b=ar.beta; alphaMap.put(wk,a); betaMap.put(wk,b)
  int n=ph.sub.getAxis().getNBins()
  for(int i=0;i<n;i++){
    double y = ph.sig.getBinContent(i) - a*ph.bkg.getBinContent(i) - b
    ph.sub.setBinContent(i, y)
  }
  System.out.printf(Locale.ROOT,"alpha,beta %s : a=%.4f  beta=%.2f  (CTRL bins=%d S=%.1f B=%.1f)%n",
    wk.toString(), a, b, ar.nUsed, ar.Ssum, ar.Bsum)
}

// global SUB with global (a,b) for QA
for(int b=0;b<SUM_SUB.getAxis().getNBins();b++){
  double y = SUM_SIG.getBinContent(b) - A_glob*SUM_BKG.getBinContent(b) - B_glob
  SUM_SUB.setBinContent(b, y)
}

// --------------- Global MPV_ref ---------------
FitG frSum=fitGlobal(SUM_SUB, FIT_LO, FIT_HI)
double MPV_REF=Double.NaN, MPV_REF_ERR=0.0
if(frSum.ok){
  MPV_REF=frSum.mpv; MPV_REF_ERR=frSum.mpvErr
  SUM_SUB_FIT=new H1F("sum_sub_fit","",ADC_NBINS,0,ADC_MAX)
  def axS=SUM_SUB_FIT.getAxis(); int nb=axS.getNBins()
  for(int i=0;i<nb;i++){
    double x=axS.getBinCenter(i)
    SUM_SUB_FIT.setBinContent(i, frSum.amp*org.jlab.groot.math.FunctionFactory.landau(x, frSum.mpv, frSum.sig))
  }
  SUM_SUB_FIT.setLineColor(3)
  System.out.printf(Locale.ROOT,"%nGlobal SUM_SUB MPV = %.1f ± %.1f ADC (chi2/ndf = %.1f/%d)%n", frSum.mpv, frSum.mpvErr, frSum.chi2, frSum.ndf)
}else{
  System.out.println("\nGlobal SUM_SUB MPV: not fitted (stats too low)")
}

// --------------- Per-wire fits → gains ---------------
Map<WireKey,Fit1D> preFit=new LinkedHashMap<>(), postFit=new LinkedHashMap<>()
Map<WireKey,Double> gain=new LinkedHashMap<>(), gainErr=new LinkedHashMap<>()
Map<WireKey,Double> mpvCorr=new LinkedHashMap<>(), mpvCorrErr=new LinkedHashMap<>()

for(Map.Entry<WireKey,PairH> e: histMap.entrySet()){
  WireKey wk=e.getKey(); PairH ph=e.getValue()
  Fit1D fr=DO_PER_WIRE_FITS? fitLandauRange(ph.sub, FIT_LO, FIT_HI, "pre_"+wk.toString().replace(' ','_')) : new Fit1D()
  preFit.put(wk, fr)

  double g=1.0, gerr=0.0, mpc=Double.NaN, mpcErr=0.0
  if(fr.ok && MPV_REF>0){
    g=MPV_REF/fr.mpv
    double rel_ref=(MPV_REF_ERR>0? MPV_REF_ERR/MPV_REF:0), rel_wire=(fr.mpvErr>0? fr.mpvErr/fr.mpv:0)
    gerr=g*Math.sqrt(rel_ref*rel_ref + rel_wire*rel_wire)
    mpc=fr.mpv*g
    double rel_pre=(fr.mpvErr>0? fr.mpvErr/fr.mpv:0), rel_g=(gerr>0? gerr/g:0)
    mpcErr=mpc*Math.sqrt(rel_pre*rel_pre + rel_g*rel_g)
  }
  gain.put(wk,g); gainErr.put(wk,gerr); mpvCorr.put(wk,mpc); mpvCorrErr.put(wk,mpcErr)
  if(fr.ok){
    System.out.printf(Locale.ROOT,"PRE MPV (FIT) %-16s : %.1f ± %.2f  | gain=%.3f ± %.3f  | MPV_corr=%.1f ± %.2f%n",
      wk.toString(), fr.mpv, fr.mpvErr, g, gerr, mpc, mpcErr)
  }else{
    System.out.printf(Locale.ROOT,"PRE MPV (FIT) %-16s : not fitted (gain=1)\n", wk.toString())
  }
}

// build SUB after gain & post fits
for(Map.Entry<WireKey,PairH> e: histMap.entrySet()){
  WireKey wk=e.getKey(); PairH ph=e.getValue()
  double g=gain.getOrDefault(wk,1.0)
  H1F hCorr=new H1F("sub_gain_"+wk.toString().replace(' ','_'),"ADC_{corr} — "+wk.toString()+";ADC_{corr};Counts",ADC_NBINS,0,ADC_MAX)
  def ax=ph.sub.getAxis(); int n=ax.getNBins(); double x0=ax.min(), bw=(ax.max()-x0)/n
  def axN=hCorr.getAxis(); int nN=axN.getNBins()
  for(int b=0;b<n;b++){
    double c=ph.sub.getBinContent(b); if(c<=0) continue
    double x=ax.getBinCenter(b)*g
    if(x<axN.min()||x>=axN.max()) continue
    int ib=(int)((x-axN.min())/bw); if(ib<0||ib>=nN) continue
    hCorr.setBinContent(ib, hCorr.getBinContent(ib)+c)
  }
  subGainMap.put(wk,hCorr)
  Fit1D frp=DO_PER_WIRE_FITS? fitLandauRange(hCorr, FIT_LO, FIT_HI, "post_"+wk.toString().replace(' ','_')) : new Fit1D()
  postFit.put(wk, frp)
}

// --------------- 2nd pass: SIG after gain (for 2D) ---------------
if(!gain.isEmpty()){
  ProgressPrintout prog2=new ProgressPrintout(); long seen2=0
  for(String fn: files){
    HipoDataSource R=new HipoDataSource()
    try{ R.open(fn) }catch(Exception ex){ System.err.println("2nd pass open fail "+fn+" : "+ex); continue }
    while(R.hasEvent()){
      DataEvent ev; try{ ev=R.getNextEvent() }catch(Exception ex){ break }
      seen2++
      RecP e=getElectronREC(ev); if(e==null){ prog2.updateStatus(); if(MAXEV>0&&seen2>=MAXEV) break; else continue }
      double pt=Math.hypot(e.px,e.py)
      double phi_e=phiDeg(e.px,e.py)
      KFRec kf_bb=bestKF_BackToBack(ev,phi_e), kf_heavy=bestKF_Heavy(ev)
      if(kf_bb.idx<0||kf_heavy.idx<0){ prog2.updateStatus(); if(MAXEV>0&&seen2>=MAXEV) break; else continue }
      int assocId=(FORCE_TRACK_ID!=null)? FORCE_TRACK_ID.intValue() : kf_heavy.trackid
      if(assocId<0){ prog2.updateStatus(); if(MAXEV>0&&seen2>=MAXEV) break; else continue }
      AssocSets AS=buildAssocSetsForTrackId(ev,assocId)
      double W=W_from_e(EBEAM,e); double W2=(Double.isNaN(W)? Double.NaN: W*W)
      boolean okW=(!Double.isNaN(W2)&&between(W2,W2_MIN,W2_MAX))
      boolean okDP=(Math.abs(dphi0to360(phi_e,kf_bb.phiDeg)-180.0)<=DPHI_HALF)

      boolean ELASTIC= okW && okDP && (!BANANA_ON || {
        long sumTRK=0L
        if(ev.hasBank("AHDC::adc")){
          DataBank a0=ev.getBank("AHDC::adc")
          for(int i=0;i<a0.rows();i++){
            int A=0; try{A=a0.getInt("ADC",i)}catch(Exception ignore){}
            if(A<=0) continue
            int Lraw=a0.getInt("layer",i), c=a0.getInt("component",i)
            int sl=Lraw/10, l=Lraw%10
            if(AS.slw.contains(new SLW(sl,l,c))) sumTRK+=(long)A
          }
        }
        inBand(pt,sumTRK,BAN_C0,BAN_C1,BAN_HALF,BAN_SCALE)
      })
      if(!ELASTIC){ prog2.updateStatus(); if(MAXEV>0&&seen2>=MAXEV) break; else continue }

      if(ev.hasBank("AHDC::adc")){
        DataBank a=ev.getBank("AHDC::adc")
        Set<String> wfGood=wfExplicitGood(ev)
        for(int i=0;i<a.rows();i++){
          int A=0; try{A=a.getInt("ADC",i)}catch(Exception ignore){}
          if(A<=0) continue
          if(!wfPassForAdcRow(a,i,wfGood)) continue
          int Lraw=a.getInt("layer",i), c=a.getInt("component",i), s=a.getInt("sector",i)
          int sl=Lraw/10, l=Lraw%10
          boolean trkPass=(assocMode==AssocMode.EXACT_WIRE)? AS.slw.contains(new SLW(sl,l,c)) : AS.sl.contains(new SL(sl,l))
          if(!trkPass) continue
          WireKey wk=new WireKey(s,Lraw,c)
          double g=gain.getOrDefault(wk,1.0)
          double Ac=(double)A*g
          if(Ac>=ADC_MIN_2D && Ac<ADC_MAX_2D) PTADC_SIG_GAIN.fill(pt,Ac)
          SUM_SIG_GAIN_1D.fill(Ac)
        }
      }
      prog2.updateStatus()
      if(MAXEV>0&&seen2>=MAXEV) break
    }
    try{ R.close() }catch(Exception ignore){}
  }
}

// --------------- Graphs + GUI ---------------
ArrayList<WireKey> keys=new ArrayList<>(histMap.keySet())
Collections.sort(keys,{a,b-> a.s!=b.s? a.s-b.s : (a.Lraw!=b.Lraw? a.Lraw-b.Lraw : a.c-b.c)} as Comparator)

int total=keys.size(), pages=(int)Math.ceil(total/(double)PAGE_SIZE())
ArrayList<String> tabs=new ArrayList<>()
for(int p=1;p<=pages;p++){
  tabs.add(String.format("SIG [p%d/%d]",p,pages))
  tabs.add(String.format("BKG [p%d/%d]",p,pages))
  tabs.add(String.format("SUB [p%d/%d]",p,pages))
  tabs.add(String.format("SUB gain [p%d/%d]",p,pages))
}
tabs.addAll(Arrays.asList("2D before","2D after gain","SUM 1D","Wire MPV & Gain","MPV PRE vs CORR"))
EmbeddedCanvasTabbed canv=new EmbeddedCanvasTabbed(tabs.toArray(new String[0]))

def drawList={ String name, int which, int page->
  def cx=canv.getCanvas(name); cx.divide(DRAW_COLS,DRAW_ROWS)
  int st=(page-1)*PAGE_SIZE(), en=Math.min(st+PAGE_SIZE(), total), pad=0
  for(int i=st;i<en;i++){
    WireKey wk=keys.get(i); PairH p=histMap.get(wk); cx.cd(pad)
    H1F h=(which==0)? p.sig : (which==1? p.bkg : p.sub)
    h.setLineColor(which==0?1:(which==1?2:4)); cx.draw(h)
    if(which==2 && DO_PER_WIRE_FITS && DRAW_GREEN_FITS){
      Fit1D fr=preFit.get(wk); if(fr!=null && fr.ok && fr.curve!=null) cx.draw(fr.curve,"same")
    }
    if(which==3){
      H1F hCorr=subGainMap.get(wk); if(hCorr!=null){ hCorr.setLineColor(2); cx.draw(hCorr) }
      Fit1D frp=postFit.get(wk); if(frp!=null && frp.ok && frp.curve!=null) cx.draw(frp.curve,"same")
    }
    YSym yr=symmetricY(h,1.15); setPadRanges(cx,pad,h.getAxis().min(),h.getAxis().max(),yr.ymin,yr.ymax); pad++
  }
}
for(int p=1;p<=pages;p++){
  drawList(String.format("SIG [p%d/%d]",p,pages),0,p)
  drawList(String.format("BKG [p%d/%d]",p,pages),1,p)
  drawList(String.format("SUB [p%d/%d]",p,pages),2,p)
  drawList(String.format("SUB gain [p%d/%d]",p,pages),3,p)
}

def c2d1=canv.getCanvas("2D before"); c2d1.divide(3,1); c2d1.cd(0); c2d1.draw(PTADC_BEFORE); c2d1.cd(1); c2d1.draw(PTADC_SIG); c2d1.cd(2); c2d1.draw(PTADC_BKG)
def c2d2=canv.getCanvas("2D after gain"); c2d2.divide(1,1); c2d2.cd(0); c2d2.draw(PTADC_SIG_GAIN)

def csum=canv.getCanvas("SUM 1D"); csum.divide(1,1); csum.cd(0); styleSum(); csum.draw(SUM_SIG); csum.draw(SUM_BKG); csum.draw(SUM_SUB); if(SUM_SUB_FIT!=null) csum.draw(SUM_SUB_FIT,"same")

GraphErrors gPre=new GraphErrors("MPV_pre"), gCorr=new GraphErrors("MPV_corr"), gGain=new GraphErrors("Gain")
int idx=0
for(WireKey wk: keys){
  Fit1D fr=preFit.get(wk); Double g=gain.get(wk), ge=gainErr.get(wk), mc=mpvCorr.get(wk), mce=mpvCorrErr.get(wk)
  if(fr!=null && fr.ok) gPre.addPoint(idx, fr.mpv, 0.0, Math.max(0.0, fr.mpvErr))
  if(mc!=null && !mc.isNaN()) gCorr.addPoint(idx, mc, 0.0, Math.max(0.0, mce?:0.0))
  if(g!=null) gGain.addPoint(idx, g, 0.0, Math.max(0.0, ge?:0.0))
  idx++
}
gPre.setTitle("MPV(pre, FIT) vs wire;wire index;ADC"); gCorr.setTitle("MPV(corr, FIT) vs wire;wire index;ADC"); gGain.setTitle("Gain vs wire;wire index;gain")
[gPre,gCorr,gGain].each{ it.setMarkerStyle(2); it.setMarkerSize(4) }
gPre.setLineColor(1); gCorr.setLineColor(2); gGain.setLineColor(3)

def cwg=canv.getCanvas("Wire MPV & Gain"); cwg.divide(2,2); cwg.cd(0); cwg.draw(gPre); cwg.cd(1); cwg.draw(gCorr); cwg.cd(2); cwg.draw(gGain)
try{ cwg.getPad(2).getAxisFrame().getAxisY().setRange(0.7,1.3) }catch(Exception ignore){}

def ccmp=canv.getCanvas("MPV PRE vs CORR"); ccmp.divide(1,1); ccmp.cd(0); ccmp.draw(gPre); ccmp.draw(gCorr,"same")

// --------------- Summary ---------------
int totalW=keys.size()
System.out.println("\n---------------- SUMMARY ----------------")
System.out.printf(Locale.ROOT,"Files                    : %d%n", files.size())
System.out.printf(Locale.ROOT,"Events processed (pass1) : %d%n", seen)
System.out.printf(Locale.ROOT,"Wires (histos built)     : %d%n", totalW)
System.out.printf(Locale.ROOT,"pT slice                 : [%.2f, %.2f] GeV%n", PT_SLICE_MIN, PT_SLICE_MAX)
System.out.printf(Locale.ROOT,"W^2 window (SIG)         : [%.3f, %.3f] GeV^2%n", W2_MIN, W2_MAX)
System.out.printf(Locale.ROOT,"Δφ window (SIG)          : |Δφ−180| ≤ %.1f°%n", DPHI_HALF)
System.out.printf(Locale.ROOT,"Banana (mode_valid)      : %s%n", BANANA_ON? "ON":"OFF")
System.out.printf(Locale.ROOT,"Proton veto (SIG)        : %s [%.0f,%.0f] ADC%n", PROTON_VETO_ON? "ON":"OFF", PROT_VETO_MIN, PROT_VETO_MAX)
System.out.printf(Locale.ROOT,"CTRL fit range           : [%.0f, %.0f] ADC (pedestal<%.0f excluded)%n", CTRL_LO, CTRL_HI, PEDESTAL_EXCLUDE_MAX)
System.out.printf(Locale.ROOT,"Global (a,b)             : a=%.4f, b=%.2f (rows=%d S=%.1f B=%.1f)%n", arSUM.a0, arSUM.beta, arSUM.nUsed, arSUM.Ssum, arSUM.Bsum)
if(!Double.isNaN(MPV_REF)) System.out.printf(Locale.ROOT,"Global MPV_ref (SUM_SUB) : %.1f ± %.1f ADC%n", MPV_REF, MPV_REF_ERR)
System.out.println("-----------------------------------------")

// --------------- Main windows (fixed constructors) ---------------
JFrame fMain=new JFrame(String.format(
  "Per-wire ADC — ALL WIRES (pT [%.2f,%.2f], CTRL=[%.0f,%.0f]) — pages=%d (grid %dx%d)",
  PT_SLICE_MIN, PT_SLICE_MAX, CTRL_LO, CTRL_HI, pages, DRAW_COLS, DRAW_ROWS))
fMain.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
fMain.setSize(1850,1100); fMain.add(canv); fMain.setLocationRelativeTo(null); fMain.setVisible(true)

// Global before/after gain (SIG)
EmbeddedCanvasTabbed canvGain=new EmbeddedCanvasTabbed("Global ADC before/after gain")
def cg=canvGain.getCanvas("Global ADC before/after gain"); cg.divide(1,1); SUM_SIG.setLineColor(1); SUM_SIG_GAIN_1D.setLineColor(2); cg.cd(0); cg.draw(SUM_SIG); cg.draw(SUM_SIG_GAIN_1D,"same")
JFrame fGain=new JFrame("Global ADC (SIG) — before vs after gain"); fGain.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); fGain.setSize(900,600); fGain.add(canvGain); fGain.setLocationRelativeTo(null); fGain.setVisible(true)
