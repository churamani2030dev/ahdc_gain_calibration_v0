// Per-wire ADC with background subtraction, robust Landau fits, gain extraction,
// re-plotted SUB histograms after gain calibration, pT(e') vs ADC (SUB) before/after gain,
// and MPV vs wire (pre & post gain) with error bars + separate overlay GUI.
//
// Usage example:
//   run-groovy gain_alert_MPV_fit_robust.groovy process \
//     -trackid 1 -nevent 5000000000 -mode_valid banana \
//     -ptmin 0.24 -ptmax 0.29 -ctrlLo 80 -ctrlHi 480 files.hipo ...

import java.util.*
import javax.swing.JFrame
import java.io.File

import org.jlab.io.hipo.HipoDataSource
import org.jlab.io.base.DataEvent
import org.jlab.io.base.DataBank

import org.jlab.groot.data.H1F
import org.jlab.groot.data.H2F
import org.jlab.groot.data.GraphErrors
import org.jlab.groot.graphics.EmbeddedCanvasTabbed
import org.jlab.jnp.utils.options.OptionStore
import org.jlab.jnp.utils.benchmark.ProgressPrintout
import org.jlab.groot.math.F1D
import org.jlab.groot.fitter.DataFitter
import groovy.transform.Field

// ------------------------------ Config ------------------------------
@Field double EBEAM      = 2.24d
@Field double M_D        = 1.875612d
@Field double W2_MIN     = 3.46d
@Field double W2_MAX     = 3.67d
@Field double DPHI_HALF  = 10.0d

// Banana (track-only ΣADC); -mode_valid banana turns this ON
@Field boolean BANANA_ON = false
@Field double BAN_SCALE  = 1000.0d
@Field double BAN_C0     = 6.0d
@Field double BAN_C1     = -35.0d
@Field double BAN_HALF   = 1.5d

// KF quality
@Field int    KF_NHITS_MIN = 8
@Field double KF_CHI2_MAX  = 30.0d

// e' fiducial
@Field double VZ_MIN  = -20.0d
@Field double VZ_MAX  = +10.0d
@Field boolean FD_ONLY= true

// pT slice (also used for 2D axes range)
@Field double PT_SLICE_MIN = 0.15d
@Field double PT_SLICE_MAX = 0.45d

@Field long   MAXEV = -1L

// Histos
@Field int    ADC_NBINS     = 220
@Field double ADC_MAX       = 4500.0d

// 2D pT vs ADC binning (X=pT, Y=ADC)
@Field int    PT_NBINS_2D   = 100
@Field double PT_MIN_2D     = 0.20d
@Field double PT_MAX_2D     = 0.45d
@Field int    ADC_NBINS_2D  = 220
@Field double ADC_MIN_2D    = 0.0d
@Field double ADC_MAX_2D    = 4500.0d

// Drawing pagination (15 pads per canvas: 5×3)
@Field int DRAW_COLS = 5
@Field int DRAW_ROWS = 3
int PAGE_SIZE(){ return DRAW_COLS*DRAW_ROWS }

// Association mode
enum AssocMode { EXACT_WIRE, LAYER_ONLY }
@Field AssocMode assocMode = AssocMode.EXACT_WIRE

// Proton veto on SIG fill (can be turned OFF)
@Field boolean PROTON_VETO_ON = false
@Field double PROT_VETO_MIN   = 100.0d
@Field double PROT_VETO_MAX   = 650.0d

// -------- Control-region α (background-only window) --------
@Field double CTRL_LO = 80.0d
@Field double CTRL_HI = 480.0d
@Field double ALPHA_MIN = 0.001d
@Field double ALPHA_MAX = 5.0d

@Field int    CTRL_MIN_BINS   = 4      // at least this many overlapping bins
@Field double CTRL_MIN_BKG_SUM= 20.0d  // at least this many BKG counts

// Optional: force KF trackid
@Field Integer FORCE_TRACK_ID = null

// -------------------------- Helpers ---------------------------------
static double deg0to360(double a){ double x=a%360.0d; return (x<0.0d)? x+360.0d : x }
static double phiDeg(float px,float py){ return deg0to360(Math.toDegrees(Math.atan2((double)py,(double)px))) }
static double dphi0to360(double pe,double pk){ double x=((pe - pk)%360.0d); return (x<0.0d)? x+360.0d : x }
static boolean between(double x,double lo,double hi){ return (x>=lo && x<=hi) }
static int Lenc(int sl,int l){ return 10*sl + l }
static double clampADC(int A, double max){ return Math.min(max-1e-6, (double)A) }
static boolean inBand(double x, double a, double b){ return x>=a && x<=b }

// Banana on track-sum only
boolean inBanana(double pt, long sumADC_trk){
  double y = ((double)sumADC_trk)/BAN_SCALE
  double c = BAN_C0 + BAN_C1*(pt - 0.26d)
  return Math.abs(y - c) <= BAN_HALF
}

// symmetric y helper
class YSym { double ymin, ymax; YSym(double a,double b){ ymin=a; ymax=b } }
YSym symmetricY(H1F h, double pad){
  int n=h.getAxis().getNBins()
  double ymin=0, ymax=0
  for(int b=0;b<n;b++){
    double y=h.getBinContent(b)
    if(b==0){ ymin=y; ymax=y }
    else { if(y<ymin) ymin=y; if(y>ymax) ymax=y }
  }
  double a = Math.max(Math.abs(ymin), Math.abs(ymax))
  a = (a<=0? 1.0 : a*pad)
  return new YSym(-a, +a)
}

void setPadRangesSafe(def canvas, int padIndex, double xmin, double xmax, double ymin, double ymax){
  try {
    canvas.getPad(padIndex).setAxisRange(xmin, xmax, ymin, ymax); return
  } catch(Throwable t) {}
  try {
    def pad = canvas.getPad(padIndex)
    def af  = pad.getAxisFrame()
    try { af.getAxisX().setRange(xmin, xmax) } catch(Throwable t1) {}
    try { af.getAxisY().setRange(ymin, ymax) } catch(Throwable t2) {}
  } catch(Throwable t) {}
}

double clampAlpha(double a){
  if(Double.isNaN(a) || Double.isInfinite(a)) return ALPHA_MIN
  return Math.max(ALPHA_MIN, Math.min(ALPHA_MAX, a))
}

// ----------------------- Data structs -------------------------------
class RecP { int pid; float px,py,pz; float vx,vy,vz,vt; byte charge; float beta,chi2pid; short status }
class KFRec { int idx=-1; float px,py; float chi2; int nhits; double phiDeg=Double.NaN; int trackid=-1 }

final class SLW {
  final int sl,l,w
  SLW(int a,int b,int c){ sl=a; l=b; w=c }
  int hashCode(){ return ((sl*1315423911) ^ (l*2654435761)) ^ w }
  boolean equals(Object o){ if(!(o instanceof SLW)) return false; SLW x=(SLW)o; return x.sl==sl && x.l==l && x.w==w }
}
final class SL {
  final int sl,l
  SL(int a,int b){ sl=a; l=b }
  int hashCode(){ return (sl*1315423911) ^ l }
  boolean equals(Object o){ if(!(o instanceof SL)) return false; SL x=(SL)o; return x.sl==sl && x.l==l }
}
final class WireKey {
  final int s,Lraw,c
  WireKey(int s,int Lraw,int c){ this.s=s; this.Lraw=Lraw; this.c=c }
  int hashCode(){ return (s*73856093) ^ (Lraw*19349663) ^ (c*83492791) }
  boolean equals(Object o){ if(!(o instanceof WireKey)) return false; WireKey k=(WireKey)o; return k.s==s && k.Lraw==Lraw && k.c==c }
  String toString(){ return String.format("S%d L%02d C%d",s,Lraw,c) }
}

// hist pair for SIG/BKG/SUB
final class PairH {
  final H1F sig, bkg, sub
  PairH(String tag, String titleBase, int nb, double lo, double hi){
    sig = new H1F("sig_"+tag, titleBase+" (SIG);ADC;Counts", nb, lo, hi)
    bkg = new H1F("bkg_"+tag, titleBase+" (BKG);ADC;Counts", nb, lo, hi)
    sub = new H1F("sub_"+tag, titleBase+" (SUB = SIG − α·BKG);ADC;Counts", nb, lo, hi)
  }
  void style(){ sig.setLineColor(1); bkg.setLineColor(2); sub.setLineColor(4) }
}

// --------------------------- Readers --------------------------------
RecP getElectronREC(DataEvent ev){
  if(!ev.hasBank("REC::Particle")) return null
  DataBank b = ev.getBank("REC::Particle")
  int best=-1
  for(int i=0;i<b.rows();i++){
    if(b.getInt("pid",i)!=11) continue
    float vz=b.getFloat("vz",i)
    short st=b.getShort("status",i)
    if(vz<VZ_MIN || vz>VZ_MAX) continue
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
  double Ee=Math.sqrt((double)e.px*e.px + (double)e.py*e.py + (double)e.pz*e.pz)
  double qx=-(double)e.px, qy=-(double)e.py, qz=Ebeam-(double)e.pz, q0=Ebeam-Ee
  double Eh=M_D+q0
  double w2=Eh*Eh - (qx*qx+qy*qy+qz*qz)
  return (w2>0.0d)? Math.sqrt(w2): Double.NaN
}

KFRec bestKF_BackToBack(DataEvent ev, double phi_e){
  KFRec out=new KFRec()
  if(!ev.hasBank("AHDC::kftrack")) return out
  DataBank k=ev.getBank("AHDC::kftrack")
  double bestAbs=Double.POSITIVE_INFINITY
  for(int i=0;i<k.rows();i++){
    int nh=k.getInt("n_hits",i); if(nh<KF_NHITS_MIN) continue
    float chi2=k.getFloat("chi2",i); if(!Float.isNaN(chi2) && (double)chi2>KF_CHI2_MAX) continue
    float px=k.getFloat("px",i), py=k.getFloat("py",i)
    double pk=phiDeg(px,py)
    double dphi=Math.abs(dphi0to360(phi_e,pk)-180.0d)
    if(dphi<bestAbs){
      bestAbs=dphi; out.idx=i; out.px=px; out.py=py; out.chi2=chi2; out.nhits=nh; out.phiDeg=pk; out.trackid=k.getInt("trackid",i)
    }
  }
  return out
}

KFRec bestKF_Heavy(DataEvent ev){
  KFRec out=new KFRec()
  if(!ev.hasBank("AHDC::kftrack")) return out
  DataBank k=ev.getBank("AHDC::kftrack")
  double best=-1.0
  for(int i=0;i<k.rows();i++){
    int nh=k.getInt("n_hits",i); if(nh<KF_NHITS_MIN) continue
    float chi2=k.getFloat("chi2",i); if(!Float.isNaN(chi2) && chi2>KF_CHI2_MAX) continue
    int sadc=0; try{ sadc=k.getInt("sum_adc",i) }catch(Exception ignore){}
    double score = (nh>0? (double)sadc/nh : -1.0)
    if(score>best){
      best=score; out.idx=i; out.nhits=nh; out.chi2=chi2
      out.px=k.getFloat("px",i); out.py=k.getFloat("py",i); out.phiDeg=phiDeg(out.px,out.py); out.trackid=k.getInt("trackid",i)
    }
  }
  return out
}

// ------------------------ WF gating ---------------------------------
Set<String> wfExplicitGood(DataEvent ev){
  HashSet<String> good=new HashSet<String>()
  if(!ev.hasBank("AHDC::wf")) return good
  DataBank w=ev.getBank("AHDC::wf")
  for(int i=0;i<w.rows();i++){
    int flag; try{ flag=w.getInt("flag",i) }catch(Exception e){ continue }
    if(flag!=0 && flag!=1) continue
    int s, LencVal, c
    try{
      int sl=w.getInt("superlayer",i), l=w.getInt("layer",i)
      s=w.getInt("sector",i); c=w.getInt("component",i)
      LencVal=Lenc(sl,l)
    }catch(Exception e){
      try{
        s=w.getInt("sector",i); c=w.getInt("component",i)
        int Lraw=w.getInt("layer",i); LencVal=Lraw
      }catch(Exception ee){ continue }
    }
    good.add(s+"#"+LencVal+"#"+c)
  }
  return good
}

boolean wfPassForAdcRow(DataBank a, int i, Set<String> wfGoodSet){
  int s=a.getInt("sector",i), Lraw=a.getInt("layer",i), c=a.getInt("component",i)
  if(wfGoodSet.contains(s+"#"+Lraw+"#"+c)) return true
  int wft = Integer.MIN_VALUE
  try { wft = a.getInt("wfType", i) } catch(Exception ignore){}
  if(wft != Integer.MIN_VALUE && wft > 2) return false
  Double tot=null
  try{ tot = (double)a.getFloat("timeOverThreshold", i) }catch(Exception ignore){}
  if(tot!=null && (tot < 250.0 || tot > 1200.0)) return false
  return true
}

// ---------------------- Track association ---------------------------
class AssocSets { Set<SLW> slw=new HashSet<SLW>(); Set<SL> sl=new HashSet<SL>() }
AssocSets buildAssocSetsForTrackId(DataEvent ev, int wantedId){
  AssocSets as=new AssocSets()
  if(!ev.hasBank("AHDC::hits")) return as
  DataBank h=ev.getBank("AHDC::hits")
  for(int i=0;i<h.rows();i++){
    int tid=h.getInt("trackid", i)
    if(tid!=wantedId) continue
    int sl=(h.getByte("superlayer",i) & 0xFF)
    int l =(h.getByte("layer",i) & 0xFF)
    int w = h.getInt("wire", i)
    as.slw.add(new SLW(sl,l,w))
    as.sl.add(new SL(sl,l))
  }
  return as
}

// ----------------------- Hist containers ----------------------------
@Field Map<WireKey,PairH>   histMap    = new LinkedHashMap<WireKey,PairH>()
@Field Map<WireKey,H1F>     fitMap     = new LinkedHashMap<WireKey,H1F>()   // visible fit curves (pre-gain SUB)
@Field Map<WireKey,H1F>     subGainMap = new LinkedHashMap<WireKey,H1F>()   // SUB after gain (rebinned)
@Field Map<WireKey,Double>  alphaMap   = new LinkedHashMap<WireKey,Double>()// per-wire α

PairH getPair(WireKey k){
  PairH p=histMap.get(k)
  if(p==null){
    String tag = k.toString().replace(' ','_')
    String ttl = "ADC — "+k.toString()+"  (p_{T} slice)"
    p = new PairH(tag, ttl, ADC_NBINS, 0.0, ADC_MAX)
    p.style()
    histMap.put(k,p)
  }
  return p
}

// ΣADC (summed over all wires) for QA
@Field H1F SUM_SIG = new H1F("sum_sig", "Per-wire ADC (SIG, all wires);ADC;Counts", ADC_NBINS, 0.0, ADC_MAX)
@Field H1F SUM_BKG = new H1F("sum_bkg", "Per-wire ADC (BKG, all wires);ADC;Counts", ADC_NBINS, 0.0, ADC_MAX)
@Field H1F SUM_SUB = new H1F("sum_sub", "Per-wire ADC (SUB=SIG−α·BKG, all wires);ADC;Counts", ADC_NBINS, 0.0, ADC_MAX)
@Field H1F SUM_SIG_GAIN_1D = new H1F("sum_sig_gain", "Per-wire ADC (SIG, all wires, gain-corr);ADC_{corr};Counts", ADC_NBINS, 0.0, ADC_MAX)
void styleSum(){ SUM_SIG.setLineColor(1); SUM_BKG.setLineColor(2); SUM_SUB.setLineColor(4) }

// Visible fit for global SUM_SUB
@Field H1F SUM_SUB_FIT = null

// ---------- 2D pT vs ADC (raw) ----------
@Field H2F PTADC_BEFORE = new H2F("ptadc_before", "ADC vs p_{T}(e') — BEFORE cuts; p_{T} (GeV); ADC",
                                  PT_NBINS_2D, PT_MIN_2D, PT_MAX_2D,
                                  ADC_NBINS_2D, ADC_MIN_2D, ADC_MAX_2D)
@Field H2F PTADC_SIG    = new H2F("ptadc_sig",    "ADC vs p_{T}(e') — SIG (all cuts); p_{T} (GeV); ADC",
                                  PT_NBINS_2D, PT_MIN_2D, PT_MAX_2D,
                                  ADC_NBINS_2D, ADC_MIN_2D, ADC_MAX_2D)
@Field H2F PTADC_BKG    = new H2F("ptadc_bkg",    "ADC vs p_{T}(e') — BKG; p_{T} (GeV); ADC",
                                  PT_NBINS_2D, PT_MIN_2D, PT_MAX_2D,
                                  ADC_NBINS_2D, ADC_MIN_2D, ADC_MAX_2D)

// ---------- 2D pT vs ADC (gain-corrected SIG only) ----------
@Field H2F PTADC_SIG_GAIN = new H2F("ptadc_sig_gain",
                                   "Gain-corrected ADC vs p_{T}(e') — SIG; p_{T} (GeV); ADC_{corr}",
                                   PT_NBINS_2D, PT_MIN_2D, PT_MAX_2D,
                                   ADC_NBINS_2D, ADC_MIN_2D, ADC_MAX_2D)

// ---------- 2D pT vs ADC (SUB, before/after gain) ----------
@Field double[][] PTADC_SUB_ARRAY       = new double[PT_NBINS_2D][ADC_NBINS_2D]
@Field double[][] PTADC_SUB_GAIN_ARRAY  = new double[PT_NBINS_2D][ADC_NBINS_2D]
@Field H2F PTADC_SUB = new H2F("ptadc_sub",
                               "ADC (SUB) vs p_{T}(e') — before gain; p_{T} (GeV); ADC",
                               PT_NBINS_2D, PT_MIN_2D, PT_MAX_2D,
                               ADC_NBINS_2D, ADC_MIN_2D, ADC_MAX_2D)
@Field H2F PTADC_SUB_GAIN = new H2F("ptadc_sub_gain",
                                    "ADC (SUB) vs p_{T}(e') — gain-corrected; p_{T} (GeV); ADC_{corr}",
                                    PT_NBINS_2D, PT_MIN_2D, PT_MAX_2D,
                                    ADC_NBINS_2D, ADC_MIN_2D, ADC_MAX_2D)

// ----------------------- α from control window ----------------------
class AlphaCtrlRes {
  double alphaLSQ
  double alphaRatio
  int    nBins
  double Ssum
  double Bsum
}

AlphaCtrlRes alphaFromControl(H1F sig, H1F bkg, double amin, double amax){
  def ax=sig.getAxis(); int n=ax.getNBins()
  double xmin=ax.min(), xmax=ax.max(), w=(xmax-xmin)/n

  double SB=0, BB=0; int used=0
  double Ssum=0, Bsum=0

  for(int b=0;b<n;b++){
    double xL=xmin+b*w, xR=xL+w
    if(xR<=amin || xL>=amax) continue

    double S=sig.getBinContent(b)
    double B=bkg.getBinContent(b)
    if(B<=0 && S<=0) continue

    double wt=1.0/Math.max(1.0, S+B)
    SB+=wt*S*B
    BB+=wt*B*B
    used++

    Ssum+=S; Bsum+=B
  }

  AlphaCtrlRes out=new AlphaCtrlRes()
  out.alphaLSQ   = (BB>0? SB/BB : 0.0)
  out.alphaRatio = (Bsum>0? Ssum/Bsum : 0.0)
  out.nBins      = used
  out.Ssum       = Ssum
  out.Bsum       = Bsum
  return out
}

// ----------------------- Landau fit helper (ROBUST) -----------------
class FitResult {
  boolean ok=false
  double amp=0, mpv=0, sigma=0
  double ampErr=0, mpvErr=0, sigErr=0
  double chi2=0
  int    ndf=0
}

FitResult fitDeuteronPeak(H1F h, double xmin, double xmax){
  FitResult out = new FitResult()
  if(h == null) return out

  // Total entries requirement (tuneable)
  double integral = h.integral()
  if(integral < 100.0) return out

  def ax = h.getAxis()
  int nb = ax.getNBins()

  int    nPosBinsInRange = 0
  double maxBinContent   = 0.0

  for(int b = 0; b < nb; b++){
    double x = ax.getBinCenter(b)
    if(x < xmin || x > xmax) continue
    double y = h.getBinContent(b)
    if(y > 0.0){
      nPosBinsInRange++
      if(y > maxBinContent) maxBinContent = y
    }
  }

  // Require structure in the fit window
  if(nPosBinsInRange < 8 || maxBinContent < 10.0) return out

  int binMax = h.getMaximumBin()
  double mpv0 = h.getAxis().getBinCenter(binMax)
  double amp0 = h.getBinContent(binMax)
  double sig0 = Math.max(50.0, (xmax - xmin)/10.0)

  F1D f = new F1D("f","[A]*landau(x,[MPV],[SIG])", xmin, xmax)
  f.setParameter(0, amp0)
  f.setParameter(1, mpv0)
  f.setParameter(2, sig0)

  try{
    DataFitter.fit(f, h, "Q") // quiet fit
  }catch(Exception ex){
    return out
  }

  out.amp   = f.getParameter(0)
  out.mpv   = f.getParameter(1)
  out.sigma = f.getParameter(2)
  try{
    out.ampErr = f.getParError(0)
    out.mpvErr = f.getParError(1)
    out.sigErr = f.getParError(2)
  }catch(Exception ignore){}

  // Manual chi2 over bins in the fit range with Poisson errors
  double chi2=0
  int npts=0
  for(int b=0;b<nb;b++){
    double x = ax.getBinCenter(b)
    if(x < xmin || x > xmax) continue
    double y = h.getBinContent(b)
    if(y <= 0) continue
    double yexp = f.evaluate(x)
    double err  = Math.sqrt(Math.max(1.0, y))
    double pull = (y - yexp)/err
    chi2 += pull*pull
    npts++
  }
  int npar = f.getNPars()
  int ndf  = npts - npar

  out.chi2 = chi2
  out.ndf  = ndf

  boolean good = !Double.isNaN(out.mpv) && out.mpv > 0 &&
                 out.mpvErr > 0 && out.sigErr > 0 &&
                 ndf > 0 && chi2 > 0
  out.ok = good
  return out
}

// ----------------------------- CLI & inputs -------------------------
OptionStore opt=new OptionStore("pd7bkgfit")
opt.addCommand("process","Per-wire ADC with BKG subtraction, robust Landau fits, and gain extraction")
final def po=opt.getOptionParser("process")
po.addOption("-nevent",""); po.addOption("-beam","")
po.addOption("-w2min",""); po.addOption("-w2max","")
po.addOption("-dphiHalf",""); po.addOption("-banana","")
po.addOption("-mode_valid","")
po.addOption("-vzmin",""); po.addOption("-vzmax",""); po.addOption("-fdonly","")
po.addOption("-ptmin",""); po.addOption("-ptmax","")
po.addOption("-ctrlLo",""); po.addOption("-ctrlHi","")
po.addOption("-protVeto",""); po.addOption("-protMin",""); po.addOption("-protMax","")
po.addOption("-trackid","")
opt.parse(args)
if(opt.getCommand()!="process"){
  System.err.println("Usage: run-groovy gain_alert_MPV_fit_robust.groovy process [opts] files.hipo ...")
  System.exit(1)
}

// glob helper
List<String> expandGlob(String pat){
  ArrayList<String> out=new ArrayList<String>()
  if(pat==null) return out
  File f=new File(pat)
  if(f.exists() && f.isFile()){ out.add(f.getPath()); return out }
  if(pat.indexOf('*')>=0 || pat.indexOf('?')>=0){
    File parent=f.getParentFile(); if(parent==null) parent=new File(".")
    String rx="\\Q"+f.getName().replace("?", "\\E.\\Q").replace("*","\\E.*\\Q")+"\\E"
    def re=java.util.regex.Pattern.compile(rx)
    File[] list=parent.listFiles()
    if(list!=null){
      for(File ff: list){
        if(ff.isFile() && re.matcher(ff.getName()).matches()) out.add(ff.getPath())
      }
    }
  }
  return out
}

// collect files
ArrayList<String> rawInputs=new ArrayList<String>()
for(String s : po.getInputList()){ rawInputs.add(s) }
ArrayList<String> files=new ArrayList<String>()
for(String s : rawInputs){
  if(s==null) continue
  if(s.toLowerCase(Locale.ROOT).endsWith(".hipo")) files.add(s)
  else if(s.indexOf('*')>=0 || s.indexOf('?')>=0) files.addAll(expandGlob(s))
}
LinkedHashSet<String> uniq=new LinkedHashSet<String>(files)
files.clear(); files.addAll(uniq)
Iterator<String> itf = files.iterator()
while(itf.hasNext()){
  String f=itf.next()
  if(!(new File(f).isFile())) itf.remove()
}
if(files.isEmpty()){ System.err.println("No .hipo inputs."); System.exit(1) }

// options
String v
try{ v=po.getOption("-nevent")?.getValue(); if(v!=null) MAXEV=Long.parseLong(v.trim()) }catch(Exception ignore){}
try{ v=po.getOption("-beam")?.getValue(); if(v!=null) EBEAM=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ v=po.getOption("-w2min")?.getValue(); if(v!=null) W2_MIN=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ v=po.getOption("-w2max")?.getValue(); if(v!=null) W2_MAX=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ v=po.getOption("-dphiHalf")?.getValue(); if(v!=null) DPHI_HALF=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ v=po.getOption("-banana")?.getValue(); if(v!=null) BANANA_ON=Boolean.parseBoolean(v.trim()) }catch(Exception ignore){}
try{ v=po.getOption("-mode_valid")?.getValue(); if(v!=null) BANANA_ON = v.trim().equalsIgnoreCase("banana") }catch(Exception ignore){}
try{ v=po.getOption("-vzmin")?.getValue(); if(v!=null) VZ_MIN=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ v=po.getOption("-vzmax")?.getValue(); if(v!=null) VZ_MAX=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ v=po.getOption("-fdonly")?.getValue(); if(v!=null) FD_ONLY=Boolean.parseBoolean(v.trim()) }catch(Exception ignore){}
try{ v=po.getOption("-ptmin")?.getValue(); if(v!=null) PT_SLICE_MIN=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ v=po.getOption("-ptmax")?.getValue(); if(v!=null) PT_SLICE_MAX=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ v=po.getOption("-ctrlLo")?.getValue(); if(v!=null) CTRL_LO=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ v=po.getOption("-ctrlHi")?.getValue(); if(v!=null) CTRL_HI=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ v=po.getOption("-protVeto")?.getValue(); if(v!=null) PROTON_VETO_ON=Boolean.parseBoolean(v.trim()) }catch(Exception ignore){}
try{ v=po.getOption("-protMin")?.getValue(); if(v!=null) PROT_VETO_MIN=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ v=po.getOption("-protMax")?.getValue(); if(v!=null) PROT_VETO_MAX=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ v=po.getOption("-trackid")?.getValue(); if(v!=null) FORCE_TRACK_ID=Integer.valueOf(v.trim()) }catch(Exception ignore){}

// ------------------------------ 1st Pass ----------------------------
styleSum()
ProgressPrintout prog=new ProgressPrintout()
long seen=0

for(String fn : files){
  HipoDataSource R=new HipoDataSource()
  try{ R.open(fn) }catch(Exception ex){ System.err.println("Open fail "+fn+" : "+ex); continue }

  while(R.hasEvent()){
    DataEvent ev
    try{ ev=R.getNextEvent() }catch(Exception ex){ break }
    seen++

    RecP e=getElectronREC(ev)
    if(e==null){ prog.updateStatus(); if(MAXEV>0 && seen>=MAXEV) break; else continue }

    double pt=Math.hypot((double)e.px,(double)e.py)

    double phi_e=phiDeg(e.px, e.py)
    KFRec kf_back  = bestKF_BackToBack(ev, phi_e)
    KFRec kf_heavy = bestKF_Heavy(ev)
    if(kf_back.idx<0 || kf_heavy.idx<0){ prog.updateStatus(); if(MAXEV>0 && seen>=MAXEV) break; else continue }

    int assocId = (FORCE_TRACK_ID!=null) ? FORCE_TRACK_ID.intValue() : kf_heavy.trackid
    if(assocId < 0){ prog.updateStatus(); if(MAXEV>0 && seen>=MAXEV) break; else continue }
    AssocSets AS = buildAssocSetsForTrackId(ev, assocId)

    double W = W_from_e(EBEAM, e); double W2=(Double.isNaN(W)? Double.NaN: W*W)
    boolean okW  = (!Double.isNaN(W2) && between(W2, W2_MIN, W2_MAX))
    boolean okDP = (Math.abs(dphi0to360(phi_e, kf_back.phiDeg)-180.0d) <= DPHI_HALF)

    // Track-only ΣADC for banana
    long sumTRK = 0L
    if(ev.hasBank("AHDC::adc")){
      DataBank a0 = ev.getBank("AHDC::adc")
      for(int i=0;i<a0.rows();i++){
        int A=0; try{ A=a0.getInt("ADC",i) }catch(Exception ignore){}
        if(A<=0) continue
        int Lraw=a0.getInt("layer",i), c=a0.getInt("component",i)
        int sl=Lraw/10, l=Lraw%10
        if(AS.slw.contains(new SLW(sl,l,c))) sumTRK += (long)A
      }
    }
    try{
      DataBank k=ev.getBank("AHDC::kftrack")
      int sadc=k.getInt("sum_adc", kf_heavy.idx)
      if(sadc>0) sumTRK = (long)sadc
    }catch(Exception ignore){}

    boolean okBan = (!BANANA_ON) || inBanana(pt, sumTRK)
    boolean ELASTIC = okW && okDP && okBan   // SIG vs BKG split

    if(ev.hasBank("AHDC::adc")){
      DataBank a=ev.getBank("AHDC::adc")
      Set<String> wfGoodSet = wfExplicitGood(ev)
      for(int i=0;i<a.rows();i++){
        int A=0; try{ A=a.getInt("ADC",i) }catch(Exception ignore){}
        if(A<=0) continue
        if(!wfPassForAdcRow(a,i,wfGoodSet)) continue

        int Lraw=a.getInt("layer",i), c=a.getInt("component",i)
        int sl=Lraw/10, l=Lraw%10
        int s=a.getInt("sector",i)

        boolean trkPass = (assocMode==AssocMode.EXACT_WIRE)
                          ? AS.slw.contains(new SLW(sl,l,c))
                          : AS.sl.contains(new SL(sl,l))
        if(!trkPass) continue

        PTADC_BEFORE.fill(pt, (double)A)

        WireKey wk=new WireKey(s,Lraw,c)
        PairH ph=getPair(wk)
        double x = clampADC(A, ADC_MAX)

        if(ELASTIC){
          if(!(PROTON_VETO_ON && inBand(x, PROT_VETO_MIN, PROT_VETO_MAX))){
            ph.sig.fill(x); SUM_SIG.fill(x)
            PTADC_SIG.fill(pt, x)
          }
        }else{
          ph.bkg.fill(x); SUM_BKG.fill(pt, x) // (fill in pT dimension is harmless; but keep as your prior)
          PTADC_BKG.fill(pt, x)
        }
      }
    }

    prog.updateStatus()
    if(MAXEV>0 && seen>=MAXEV) break
  }
  try{ R.close() }catch(Exception ignore){}
}

// ----------------------- α via control window -----------------------
AlphaCtrlRes arSUM = alphaFromControl(SUM_SIG, SUM_BKG, CTRL_LO, CTRL_HI)
double A_glob = arSUM.alphaLSQ>0 ? arSUM.alphaLSQ :
                (arSUM.alphaRatio>0? arSUM.alphaRatio : 0.0)
A_glob = clampAlpha(A_glob)

// wire-by-wire subtraction
for(Map.Entry<WireKey,PairH> e : histMap.entrySet()){
  WireKey wk = e.getKey()
  PairH  ph  = e.getValue()

  AlphaCtrlRes ar = alphaFromControl(ph.sig, ph.bkg, CTRL_LO, CTRL_HI)
  double aWire
  boolean useGlob=false
  boolean goodStats = (ar.nBins >= CTRL_MIN_BINS && ar.Bsum >= CTRL_MIN_BKG_SUM)

  if(goodStats){
    if(ar.alphaLSQ>0)        aWire=ar.alphaLSQ
    else if(ar.alphaRatio>0) aWire=ar.alphaRatio
    else { aWire=A_glob; useGlob=true }
  }else{
    aWire=A_glob; useGlob=true
  }
  aWire = clampAlpha(aWire)
  alphaMap.put(wk, aWire)

  int n=ph.sub.getAxis().getNBins()
  for(int b=0;b<n;b++){
    double y = ph.sig.getBinContent(b) - aWire*ph.bkg.getBinContent(b)
    ph.sub.setBinContent(b, y)
  }

  System.out.printf(Locale.ROOT,
    "alpha %s = %.4f  (LSQ=%.4f, ratio=%.4f, glob=%.4f, S=%.1f, B=%.1f, bins=%d, useGlob=%s)%n",
    wk.toString(), aWire,
    ar.alphaLSQ, ar.alphaRatio, A_glob,
    ar.Ssum, ar.Bsum, ar.nBins, useGlob?"Y":"N")
}

// global SUB
for(int b=0;b<SUM_SUB.getAxis().getNBins();b++){
  double y = SUM_SIG.getBinContent(b) - A_glob*SUM_BKG.getBinContent(b)
  SUM_SUB.setBinContent(b, y)
}

// --------------------------- Landau fits (PRE) ----------------------
@Field Map<WireKey,FitResult> mpvPreMap   = new LinkedHashMap<WireKey,FitResult>()
@Field Map<WireKey,Double>    gainMap     = new LinkedHashMap<WireKey,Double>()

@Field double FIT_LO = 450.0d
@Field double FIT_HI = 2000.0d

for(Map.Entry<WireKey,PairH> e : histMap.entrySet()){
  WireKey wk = e.getKey()
  PairH  ph  = e.getValue()
  FitResult fr = fitDeuteronPeak(ph.sub, FIT_LO, FIT_HI)
  mpvPreMap.put(wk, fr)

  if(fr!=null && fr.ok){
    System.out.printf(Locale.ROOT,
      "MPV %s : %.1f ± %.1f ADC   (chi2/ndf = %.1f/%d)%n",
      wk.toString(), fr.mpv, fr.mpvErr, fr.chi2, fr.ndf)

    H1F hf = new H1F("fit_"+wk.toString().replace(' ','_'),"",ADC_NBINS,0.0,ADC_MAX)
    def ax = hf.getAxis()
    int nb = ax.getNBins()
    for(int b=0;b<nb;b++){
      double x = ax.getBinCenter(b)
      double y = fr.amp * org.jlab.groot.math.FunctionFactory.landau(x, fr.mpv, fr.sigma)
      hf.setBinContent(b, y)
    }
    hf.setLineColor(3)
    fitMap.put(wk, hf)
  }else{
    System.out.printf(Locale.ROOT,
      "MPV %s : not fitted (too few entries or bad fit)%n", wk.toString())
  }
}

// global fit on SUM_SUB (reference MPV)
FitResult frSum = fitDeuteronPeak(SUM_SUB, FIT_LO, FIT_HI)
double MPV_REF = Double.NaN
double MPV_REF_ERR = 0.0
if(frSum!=null && frSum.ok){
  MPV_REF = frSum.mpv
  MPV_REF_ERR = frSum.mpvErr
  System.out.printf(Locale.ROOT,
    "%nGlobal SUM_SUB MPV = %.1f ± %.1f ADC  (chi2/ndf = %.1f/%d)%n",
    frSum.mpv, frSum.mpvErr, frSum.chi2, frSum.ndf)

  SUM_SUB_FIT = new H1F("sum_sub_fit","",ADC_NBINS,0.0,ADC_MAX)
  def axS = SUM_SUB_FIT.getAxis()
  int nbS = axS.getNBins()
  for(int b=0;b<nbS;b++){
    double x = axS.getBinCenter(b)
    double y = frSum.amp * org.jlab.groot.math.FunctionFactory.landau(x, frSum.mpv, frSum.sigma)
    SUM_SUB_FIT.setBinContent(b, y)
  }
  SUM_SUB_FIT.setLineColor(3)
}else{
  System.out.println("\nGlobal SUM_SUB MPV: not fitted (too few entries or bad fit)")
}

// --------------------------- Gain per wire --------------------------
if(!Double.isNaN(MPV_REF)){
  for(Map.Entry<WireKey,FitResult> e : mpvPreMap.entrySet()){
    WireKey wk = e.getKey()
    FitResult fr = e.getValue()
    double g = 1.0
    if(fr!=null && fr.ok && fr.mpv>0){
      g = MPV_REF / fr.mpv
    }
    gainMap.put(wk, g)
  }
}else{
  for(WireKey wk : histMap.keySet()) gainMap.put(wk, 1.0)
}

// ------------------ Build per-wire SUB histos after gain ------------
for(Map.Entry<WireKey,PairH> e : histMap.entrySet()){
  WireKey wk = e.getKey()
  PairH  ph  = e.getValue()
  H1F hSub   = ph.sub

  double g = gainMap.containsKey(wk) ? gainMap.get(wk) : 1.0d

  H1F hCorr = new H1F("sub_gain_"+wk.toString().replace(' ','_'),
                      "ADC_{corr} — "+wk.toString()+" (SUB gain-corr);ADC_{corr};Counts",
                      ADC_NBINS, 0.0, ADC_MAX)
  def axOld = hSub.getAxis()
  int nOld  = axOld.getNBins()
  double xMinOld = axOld.min()
  double xMaxOld = axOld.max()
  double bwOld   = (xMaxOld - xMinOld)/nOld

  def axNew = hCorr.getAxis()
  int nNew  = axNew.getNBins()
  double xMinNew = axNew.min()
  double xMaxNew = axNew.max()

  for(int b=0;b<nOld;b++){
    double c = hSub.getBinContent(b)
    if(c<=0) continue
    double x = axOld.getBinCenter(b)
    double xCorr = x * g
    if(xCorr < xMinNew || xCorr >= xMaxNew) continue
    int ibNew = (int)((xCorr - xMinNew)/bwOld)
    if(ibNew<0 || ibNew>=nNew) continue
    double cur = hCorr.getBinContent(ibNew)
    hCorr.setBinContent(ibNew, cur + c)
  }

  subGainMap.put(wk, hCorr)
}

// --------------------------- 2nd Pass: gain-corrected pT–ADC (SIG) --
if(!gainMap.isEmpty()){
  ProgressPrintout prog2 = new ProgressPrintout()
  long seen2 = 0L

  for(String fn : files){
    HipoDataSource R=new HipoDataSource()
    try{ R.open(fn) }catch(Exception ex){ System.err.println("2nd pass open fail "+fn+" : "+ex); continue }
    while(R.hasEvent()){
      DataEvent ev
      try{ ev=R.getNextEvent() }catch(Exception ex){ break }
      seen2++

      RecP e=getElectronREC(ev)
      if(e==null){ prog2.updateStatus(); if(MAXEV>0 && seen2>=MAXEV) break; else continue }

      double pt=Math.hypot((double)e.px,(double)e.py)

      double phi_e=phiDeg(e.px, e.py)
      KFRec kf_back  = bestKF_BackToBack(ev, phi_e)
      KFRec kf_heavy = bestKF_Heavy(ev)
      if(kf_back.idx<0 || kf_heavy.idx<0){ prog2.updateStatus(); if(MAXEV>0 && seen2>=MAXEV) break; else continue }

      int assocId = (FORCE_TRACK_ID!=null) ? FORCE_TRACK_ID.intValue() : kf_heavy.trackid
      if(assocId < 0){ prog2.updateStatus(); if(MAXEV>0 && seen2>=MAXEV) break; else continue }
      AssocSets AS = buildAssocSetsForTrackId(ev, assocId)

      double W = W_from_e(EBEAM, e); double W2=(Double.isNaN(W)? Double.NaN: W*W)
      boolean okW  = (!Double.isNaN(W2) && between(W2, W2_MIN, W2_MAX))
      boolean okDP = (Math.abs(dphi0to360(phi_e, kf_back.phiDeg)-180.0d) <= DPHI_HALF)

      long sumTRK = 0L
      if(ev.hasBank("AHDC::adc")){
        DataBank a0 = ev.getBank("AHDC::adc")
        for(int i=0;i<a0.rows();i++){
          int A=0; try{ A=a0.getInt("ADC",i) }catch(Exception ignore){}
          if(A<=0) continue
          int Lraw=a0.getInt("layer",i), c=a0.getInt("component",i)
          int sl=Lraw/10, l=Lraw%10
          if(AS.slw.contains(new SLW(sl,l,c))) sumTRK += (long)A
        }
      }
      try{
        DataBank k=ev.getBank("AHDC::kftrack")
        int sadc=k.getInt("sum_adc", kf_heavy.idx)
        if(sadc>0) sumTRK = (long)sadc
      }catch(Exception ignore){}

      boolean okBan = (!BANANA_ON) || inBanana(pt, sumTRK)
      boolean ELASTIC = okW && okDP && okBan

      if(!ELASTIC){ prog2.updateStatus(); if(MAXEV>0 && seen2>=MAXEV) break; else continue }

      if(ev.hasBank("AHDC::adc")){
        DataBank a=ev.getBank("AHDC::adc")
        Set<String> wfGoodSet = wfExplicitGood(ev)
        for(int i=0;i<a.rows();i++){
          int A=0; try{ A=a.getInt("ADC",i) }catch(Exception ignore){}
          if(A<=0) continue
          if(!wfPassForAdcRow(a,i,wfGoodSet)) continue

          int Lraw=a.getInt("layer",i), c=a.getInt("component",i)
          int sl=Lraw/10, l=Lraw%10
          int s=a.getInt("sector",i)

          boolean trkPass = (assocMode==AssocMode.EXACT_WIRE)
                            ? AS.slw.contains(new SLW(sl,l,c))
                            : AS.sl.contains(new SL(sl,l))
          if(!trkPass) continue

          WireKey wk = new WireKey(s,Lraw,c)
          double g = gainMap.containsKey(wk) ? gainMap.get(wk) : 1.0d
          double Acorr = ((double)A) * g
          if(Acorr < ADC_MIN_2D || Acorr>ADC_MAX_2D) continue
          PTADC_SIG_GAIN.fill(pt, Acorr)
          SUM_SIG_GAIN_1D.fill(Acorr)
        }
      }

      prog2.updateStatus()
      if(MAXEV>0 && seen2>=MAXEV) break
    }
    try{ R.close() }catch(Exception ignore){}
  }
}

// --------------------------- 3rd Pass: pT–ADC (SUB) -----------------
if(!gainMap.isEmpty() && !alphaMap.isEmpty()){
  ProgressPrintout prog3 = new ProgressPrintout()
  long seen3 = 0L

  double bwX = (PT_MAX_2D - PT_MIN_2D)/PT_NBINS_2D
  double bwY = (ADC_MAX_2D - ADC_MIN_2D)/ADC_NBINS_2D

  for(String fn : files){
    HipoDataSource R=new HipoDataSource()
    try{ R.open(fn) }catch(Exception ex){ System.err.println("3rd pass open fail "+fn+" : "+ex); continue }
    while(R.hasEvent()){
      DataEvent ev
      try{ ev=R.getNextEvent() }catch(Exception ex){ break }
      seen3++

      RecP e=getElectronREC(ev)
      if(e==null){ prog3.updateStatus(); if(MAXEV>0 && seen3>=MAXEV) break; else continue }

      double pt=Math.hypot((double)e.px,(double)e.py)

      double phi_e=phiDeg(e.px, e.py)
      KFRec kf_back  = bestKF_BackToBack(ev, phi_e)
      KFRec kf_heavy = bestKF_Heavy(ev)
      if(kf_back.idx<0 || kf_heavy.idx<0){ prog3.updateStatus(); if(MAXEV>0 && seen3>=MAXEV) break; else continue }

      int assocId = (FORCE_TRACK_ID!=null) ? FORCE_TRACK_ID.intValue() : kf_heavy.trackid
      if(assocId < 0){ prog3.updateStatus(); if(MAXEV>0 && seen3>=MAXEV) break; else continue }
      AssocSets AS = buildAssocSetsForTrackId(ev, assocId)

      double W = W_from_e(EBEAM, e); double W2=(Double.isNaN(W)? Double.NaN: W*W)
      boolean okW  = (!Double.isNaN(W2) && between(W2, W2_MIN, W2_MAX))
      boolean okDP = (Math.abs(dphi0to360(phi_e, kf_back.phiDeg)-180.0d) <= DPHI_HALF)

      // replicate banana logic
      long sumTRK = 0L
      if(ev.hasBank("AHDC::adc")){
        DataBank a0 = ev.getBank("AHDC::adc")
        for(int i=0;i<a0.rows();i++){
          int A=0; try{ A=a0.getInt("ADC",i) }catch(Exception ignore){}
          if(A<=0) continue
          int Lraw=a0.getInt("layer",i), c=a0.getInt("component",i)
          int sl=Lraw/10, l=Lraw%10
          if(AS.slw.contains(new SLW(sl,l,c))) sumTRK += (long)A
        }
      }
      try{
        DataBank k=ev.getBank("AHDC::kftrack")
        int sadc=k.getInt("sum_adc", kf_heavy.idx)
        if(sadc>0) sumTRK = (long)sadc
      }catch(Exception ignore){}
      boolean okBan = (!BANANA_ON) || inBanana(pt, sumTRK)

      boolean ELASTIC = okW && okDP && okBan

      if(ev.hasBank("AHDC::adc")){
        DataBank a=ev.getBank("AHDC::adc")
        Set<String> wfGoodSet = wfExplicitGood(ev)
        for(int i=0;i<a.rows();i++){
          int A=0; try{ A=a.getInt("ADC",i) }catch(Exception ignore){}
          if(A<=0) continue
          if(!wfPassForAdcRow(a,i,wfGoodSet)) continue

          int Lraw=a.getInt("layer",i), c=a.getInt("component",i)
          int sl=Lraw/10, l=Lraw%10
          int s=a.getInt("sector",i)

          boolean trkPass = (assocMode==AssocMode.EXACT_WIRE)
                            ? AS.slw.contains(new SLW(sl,l,c))
                            : AS.sl.contains(new SL(sl,l))
          if(!trkPass) continue

          WireKey wk = new WireKey(s,Lraw,c)
          double aWire = alphaMap.containsKey(wk) ? alphaMap.get(wk) : A_glob
          double g     = gainMap.containsKey(wk) ? gainMap.get(wk)  : 1.0d
          double weight = ELASTIC ? 1.0d : -aWire

          double x = pt
          double y = (double)A
          if(x>=PT_MIN_2D && x<PT_MAX_2D && y>=ADC_MIN_2D && y<ADC_MAX_2D){
            int ix = (int)((x-PT_MIN_2D)/bwX)
            int iy = (int)((y-ADC_MIN_2D)/bwY)
            if(ix>=0 && ix<PT_NBINS_2D && iy>=0 && iy<ADC_NBINS_2D){
              PTADC_SUB_ARRAY[ix][iy] += weight
            }
          }

          double yCorr = ((double)A)*g
          if(yCorr>=ADC_MIN_2D && yCorr<ADC_MAX_2D){
            int ix2 = (int)((x-PT_MIN_2D)/bwX)
            int iy2 = (int)((yCorr-ADC_MIN_2D)/bwY)
            if(ix2>=0 && ix2<PT_NBINS_2D && iy2>=0 && iy2<ADC_NBINS_2D){
              PTADC_SUB_GAIN_ARRAY[ix2][iy2] += weight
            }
          }
        }
      }

      prog3.updateStatus()
      if(MAXEV>0 && seen3>=MAXEV) break
    }
    try{ R.close() }catch(Exception ignore){}
  }

  // copy arrays into H2F histos
  for(int ix=0; ix<PT_NBINS_2D; ix++){
    for(int iy=0; iy<ADC_NBINS_2D; iy++){
      PTADC_SUB.setBinContent(ix, iy, PTADC_SUB_ARRAY[ix][iy])
      PTADC_SUB_GAIN.setBinContent(ix, iy, PTADC_SUB_GAIN_ARRAY[ix][iy])
    }
  }
}

// --------------------- POST-GAIN per-wire Landau fits ---------------
@Field Map<WireKey,FitResult> mpvPostMap = new LinkedHashMap<WireKey,FitResult>()

for(Map.Entry<WireKey,H1F> e : subGainMap.entrySet()){
  WireKey wk = e.getKey()
  H1F hCorr  = e.getValue()
  FitResult frG = fitDeuteronPeak(hCorr, FIT_LO, FIT_HI)
  mpvPostMap.put(wk, frG)
  if(frG!=null && frG.ok){
    System.out.printf(Locale.ROOT,
      "POST-GAIN MPV %s : %.1f ± %.1f ADC   (chi2/ndf = %.1f/%d)%n",
      wk.toString(), frG.mpv, frG.mpvErr, frG.chi2, frG.ndf)
  }else{
    System.out.printf(Locale.ROOT,
      "POST-GAIN MPV %s : not fitted (too few entries or bad fit)%n", wk.toString())
  }
}

// ----------------------------- GUI ----------------------------------
ArrayList<WireKey> allKeys = new ArrayList<WireKey>(histMap.keySet())
Collections.sort(allKeys, new Comparator<WireKey>(){
  int compare(WireKey a, WireKey b){
    if(a.s!=b.s) return a.s-b.s
    if(a.Lraw!=b.Lraw) return a.Lraw-b.Lraw
    return a.c-b.c
  }
})

int total = allKeys.size()
int pages = (int)Math.ceil(total / (double)PAGE_SIZE())
ArrayList<String> tabs=new ArrayList<String>()
for(int p=1;p<=pages;p++){
  tabs.add(String.format("Per-wire SIG [p%d/%d]", p, pages))
  tabs.add(String.format("Per-wire BKG [p%d/%d]", p, pages))
  tabs.add(String.format("Per-wire SUB [p%d/%d]", p, pages))
  tabs.add(String.format("Per-wire SUB gain [p%d/%d]", p, pages))
  tabs.add(String.format("SIG vs BKG [p%d/%d]", p, pages))
}
tabs.add("pT vs ADC (2D)")
tabs.add("pT vs ADC (gain-corr)")
tabs.add("SUM 1D (SIG/BKG/SUB)")
tabs.add("Wire MPV & Gain (PRE)")
tabs.add("Wire MPV (POST)")
tabs.add("Gain vs Wire")

EmbeddedCanvasTabbed canv = new EmbeddedCanvasTabbed(tabs.toArray(new String[0]))

def drawPage = { String name, int which, int pageIdx ->
  def cx = canv.getCanvas(name)
  cx.divide(DRAW_COLS, DRAW_ROWS)
  int start = (pageIdx-1)*PAGE_SIZE()
  int end   = Math.min(start+PAGE_SIZE(), total)
  int pad=0
  for(int i=start;i<end;i++){
    WireKey wk = allKeys.get(i)
    PairH p    = histMap.get(wk)
    cx.cd(pad)
    H1F h = (which==0)? p.sig : (which==1? p.bkg : p.sub)
    cx.draw(h)
    if(which==2){
      H1F hf = fitMap.get(wk)
      if(hf!=null) cx.draw(hf, "same")
      YSym yr = symmetricY(h, 1.15)
      double xmin=h.getAxis().min(), xmax=h.getAxis().max()
      setPadRangesSafe(cx, pad, xmin, xmax, yr.ymin, yr.ymax)
    }
    pad++
  }
}

def drawOverlaySIGBKG = { String name, int pageIdx ->
  def cx = canv.getCanvas(name)
  cx.divide(DRAW_COLS, DRAW_ROWS)
  int start = (pageIdx-1)*PAGE_SIZE()
  int end   = Math.min(start+PAGE_SIZE(), total)
  int pad=0
  for(int i=start;i<end;i++){
    WireKey wk = allKeys.get(i)
    PairH p    = histMap.get(wk)
    cx.cd(pad++)
    p.sig.setLineColor(1)
    p.bkg.setLineColor(2)
    cx.draw(p.sig)
    cx.draw(p.bkg, "same")
  }
}

// per-wire SUB (orig vs gain-corr) overlay
def drawSubGainPage = { String name, int pageIdx ->
  def cx = canv.getCanvas(name)
  cx.divide(DRAW_COLS, DRAW_ROWS)
  int start = (pageIdx-1)*PAGE_SIZE()
  int end   = Math.min(start+PAGE_SIZE(), total)
  int pad=0
  for(int i=start;i<end;i++){
    WireKey wk = allKeys.get(i)
    PairH p    = histMap.get(wk)
    H1F hOrig  = p.sub
    H1F hCorr  = subGainMap.get(wk)
    cx.cd(pad)
    hOrig.setLineColor(1)  // black: original SUB
    cx.draw(hOrig)
    if(hCorr!=null){
      hCorr.setLineColor(2) // red: SUB after gain
      cx.draw(hCorr, "same")
    }
    YSym yr = symmetricY(hOrig, 1.15)
    double xmin=hOrig.getAxis().min(), xmax=hOrig.getAxis().max()
    setPadRangesSafe(cx, pad, xmin, xmax, yr.ymin, yr.ymax)
    pad++
  }
}

// render pages
for(int p=1;p<=pages;p++){
  drawPage(String.format("Per-wire SIG [p%d/%d]", p, pages), 0, p)
  drawPage(String.format("Per-wire BKG [p%d/%d]", p, pages), 1, p)
  drawPage(String.format("Per-wire SUB [p%d/%d]", p, pages), 2, p)
  drawSubGainPage(String.format("Per-wire SUB gain [p%d/%d]", p, pages), p)
  drawOverlaySIGBKG(String.format("SIG vs BKG [p%d/%d]", p, pages), p)
}

// 2D tabs (SIG samples)
def c2d = canv.getCanvas("pT vs ADC (2D)")
c2d.divide(3,1)
c2d.cd(0); c2d.draw(PTADC_BEFORE)
c2d.cd(1); c2d.draw(PTADC_SIG)
c2d.cd(2); c2d.draw(PTADC_BKG)

def c2dg = canv.getCanvas("pT vs ADC (gain-corr)")
c2dg.divide(1,1)
c2dg.cd(0); c2dg.draw(PTADC_SIG_GAIN)

// SUM tab with fit
styleSum()
def csum = canv.getCanvas("SUM 1D (SIG/BKG/SUB)")
csum.divide(1,1); csum.cd(0)
csum.draw(SUM_SIG); csum.draw(SUM_BKG); csum.draw(SUM_SUB)
if(SUM_SUB_FIT!=null) csum.draw(SUM_SUB_FIT,"same")
YSym yrSum = symmetricY(SUM_SUB, 1.15)
double xminS=SUM_SUB.getAxis().min(), xmaxS=SUM_SUB.getAxis().max()
setPadRangesSafe(csum, 0, xminS, xmaxS, yrSum.ymin, yrSum.ymax)

// -------------- Graphs with ERROR BARS & limits ---------------------

// MPV vs wire (PRE) with error bars (plot only successful fits)
GraphErrors gMPVpre = new GraphErrors("MPV_pre_vs_wire")
gMPVpre.setTitle("MPV (PRE) vs wire;wire index;MPV (ADC)")
int idxW=0
for(WireKey wk : allKeys){
  FitResult fr = mpvPreMap.get(wk)
  if(fr!=null && fr.ok){
    gMPVpre.addPoint((double)idxW, fr.mpv, 0.0, fr.mpvErr)
  }
  idxW++
}

// MPV vs wire (POST) with error bars (only successful fits)
GraphErrors gMPVpost = new GraphErrors("MPV_post_vs_wire")
gMPVpost.setTitle("MPV (POST) vs wire;wire index;MPV (ADC)")
idxW=0
for(WireKey wk : allKeys){
  FitResult fr = mpvPostMap.get(wk)
  if(fr!=null && fr.ok){
    gMPVpost.addPoint((double)idxW, fr.mpv, 0.0, fr.mpvErr)
  }
  idxW++
}

// Gain vs wire with propagated errors from MPV_ref and MPV_pre
GraphErrors gGAIN = new GraphErrors("GAIN_vs_wire")
gGAIN.setTitle("Gain vs wire;wire index;gain (MPV_ref / MPV_pre)")
idxW=0
for(WireKey wk : allKeys){
  FitResult fr = mpvPreMap.get(wk)
  if(fr!=null && fr.ok && fr.mpv>0){
    double g = gainMap.get(wk)
    double rel2 = 0.0
    if(MPV_REF>0 && MPV_REF_ERR>0) rel2 += (MPV_REF_ERR/MPV_REF)*(MPV_REF_ERR/MPV_REF)
    if(fr.mpvErr>0) rel2 += (fr.mpvErr/fr.mpv)*(fr.mpvErr/fr.mpv)
    double gErr = g * Math.sqrt(rel2)
    gGAIN.addPoint((double)idxW, g, 0.0, gErr)
  }
  idxW++
}

// Draw MPV & Gain (PRE)
def cwgPre = canv.getCanvas("Wire MPV & Gain (PRE)")
cwgPre.divide(1,2)
cwgPre.cd(0); cwgPre.draw(gMPVpre)
try{ cwgPre.getPad(0).getAxisFrame().getAxisY().setRange(0.0, 1500.0) }catch(Exception ignore){}
cwgPre.cd(1); cwgPre.draw(gGAIN)
try{ cwgPre.getPad(1).getAxisFrame().getAxisY().setRange(0.0, 3.0) }catch(Exception ignore){}

// Draw MPV (POST)
def cwgPost = canv.getCanvas("Wire MPV (POST)")
cwgPost.divide(1,1)
cwgPost.cd(0); cwgPost.draw(gMPVpost)
try{ cwgPost.getPad(0).getAxisFrame().getAxisY().setRange(0.0, 1500.0) }catch(Exception ignore){}

// Draw Gain vs Wire (standalone tab)
def cGain = canv.getCanvas("Gain vs Wire")
cGain.divide(1,1)
cGain.cd(0); cGain.draw(gGAIN)
try{ cGain.getPad(0).getAxisFrame().getAxisY().setRange(0.0, 3.0) }catch(Exception ignore){}

// Main frame
JFrame f=new JFrame(String.format(
  "Per-wire ADC — ALL WIRES (pT [%.2f,%.2f], CTRL=[%.0f,%.0f], proton veto %s) — pages=%d (grid %dx%d)",
  PT_SLICE_MIN, PT_SLICE_MAX, CTRL_LO, CTRL_HI, PROTON_VETO_ON?"ON":"OFF", pages, DRAW_COLS, DRAW_ROWS))
f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
f.setSize(1850,1100); f.add(canv); f.setLocationRelativeTo(null); f.setVisible(true)

// -------- Separate GUI: global ADC before/after gain (SIG only) ----
if(SUM_SIG_GAIN_1D!=null){
  EmbeddedCanvasTabbed canvGain = new EmbeddedCanvasTabbed("Global ADC before/after gain")
  def cg = canvGain.getCanvas("Global ADC before/after gain")
  cg.divide(1,1)
  cg.cd(0)
  SUM_SIG.setLineColor(1)        // black
  SUM_SIG_GAIN_1D.setLineColor(2)// red
  cg.draw(SUM_SIG)
  cg.draw(SUM_SIG_GAIN_1D,"same")

  JFrame fg=new JFrame("Global ADC (SIG) — before vs after gain")
  fg.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
  fg.setSize(900,600); fg.add(canvGain); fg.setLocationRelativeTo(null); fg.setVisible(true)
}

// -------- Separate GUI: pT vs ADC (SUB) before/after gain ----------
if(PTADC_SUB!=null && PTADC_SUB_GAIN!=null){
  EmbeddedCanvasTabbed canvSub = new EmbeddedCanvasTabbed("pT vs ADC (SUB) before/after gain")
  def cs = canvSub.getCanvas("pT vs ADC (SUB) before/after gain")
  cs.divide(2,1)
  cs.cd(0); cs.draw(PTADC_SUB)       // left: SUB before gain
  cs.cd(1); cs.draw(PTADC_SUB_GAIN)  // right: SUB after gain

  JFrame fs = new JFrame("pT(e') vs ADC (SUB) — before/after gain")
  fs.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
  fs.setSize(1400,600); fs.add(canvSub); fs.setLocationRelativeTo(null); fs.setVisible(true)
}

// -------- Separate GUI: MPV overlay (PRE vs POST) with errors -------
EmbeddedCanvasTabbed canvMPVOverlay = new EmbeddedCanvasTabbed("MPV overlay (PRE vs POST)")
def cm = canvMPVOverlay.getCanvas("MPV overlay (PRE vs POST)")
cm.divide(1,1)

// Build two aligned GraphErrors only for indices where each fit succeeded
GraphErrors gMPVpreOverlay  = new GraphErrors("MPV_PRE_overlay")
GraphErrors gMPVpostOverlay = new GraphErrors("MPV_POST_overlay")
gMPVpreOverlay.setTitle("MPV overlay;wire index;MPV (ADC)")
gMPVpostOverlay.setTitle("MPV overlay;wire index;MPV (ADC)")

idxW=0
for(WireKey wk : allKeys){
  FitResult frPre  = mpvPreMap.get(wk)
  FitResult frPost = mpvPostMap.get(wk)
  if(frPre!=null && frPre.ok){
    gMPVpreOverlay.addPoint((double)idxW, frPre.mpv, 0.0, frPre.mpvErr)
  }
  if(frPost!=null && frPost.ok){
    gMPVpostOverlay.addPoint((double)idxW, frPost.mpv, 0.0, frPost.mpvErr)
  }
  idxW++
}
gMPVpreOverlay.setMarkerColor(1)   // black
gMPVpostOverlay.setMarkerColor(2)  // red

cm.cd(0); cm.draw(gMPVpreOverlay); cm.draw(gMPVpostOverlay,"same")
try{ cm.getPad(0).getAxisFrame().getAxisY().setRange(0.0, 1500.0) }catch(Exception ignore){}

JFrame fm = new JFrame("MPV vs Wire — PRE (black) vs POST (red)")
fm.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
fm.setSize(1200,600); fm.add(canvMPVOverlay); fm.setLocationRelativeTo(null); fm.setVisible(true)

// --------------------------- Summary --------------------------------
System.out.println("\n---------------- SUMMARY ----------------")
System.out.printf(Locale.ROOT, "Files                    : %d%n", files.size())
System.out.printf(Locale.ROOT, "Events processed (pass1) : %d%n", seen)
System.out.printf(Locale.ROOT, "Wires (histos built)     : %d%n", total)
System.out.printf(Locale.ROOT, "pT slice                 : [%.2f, %.2f] GeV%n", PT_SLICE_MIN, PT_SLICE_MAX)
System.out.printf(Locale.ROOT, "W² window (SIG)          : [%.3f, %.3f] GeV²%n", W2_MIN, W2_MAX)
System.out.printf(Locale.ROOT, "Δφ window (SIG)          : |Δφ−180| ≤ %.1f°%n", DPHI_HALF)
System.out.printf(Locale.ROOT, "Banana (mode_valid)      : %s%n", BANANA_ON?"ON":"OFF")
System.out.printf(Locale.ROOT, "Proton veto (SIG)        : %s [%.0f,%.0f] ADC%n", PROTON_VETO_ON?"ON":"OFF", PROT_VETO_MIN, PROT_VETO_MAX)
System.out.printf(Locale.ROOT, "CTRL (α via LSQ)         : [%.0f, %.0f] ADC%n", CTRL_LO, CTRL_HI)
System.out.printf(Locale.ROOT, "Global α (SUM)           : %.4f  (S=%.1f, B=%.1f, bins=%d)%n",
                  A_glob, arSUM.Ssum, arSUM.Bsum, arSUM.nBins)
if(!Double.isNaN(MPV_REF)){
  System.out.printf(Locale.ROOT, "Global MPV_ref (SUM_SUB) : %.1f ± %.1f ADC%n", MPV_REF, MPV_REF_ERR)
}
System.out.println("-----------------------------------------")
