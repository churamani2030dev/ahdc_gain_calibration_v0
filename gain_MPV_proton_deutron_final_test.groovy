// gain_MPV_proton_deutron_final_test.groovy
// Full pipeline: SIG/BKG/SUB with per-wire α,β; MPV extraction with errors;
// global REF (trimmed median) + error; Gain = REF/MPV with propagated errors;
// QA tabs: ΣADC, SIG, BKG, SUB, MPV, Gain.
//
// Colors: SIG=black, BKG=red, SUB=blue, overlays=green.
//
// --------------------------- Imports ---------------------------
import java.util.*
import javax.swing.JFrame
import groovy.transform.Field

import org.jlab.io.hipo.HipoDataSource
import org.jlab.io.base.DataEvent
import org.jlab.io.base.DataBank

import org.jlab.groot.data.H1F
import org.jlab.groot.data.H2F
import org.jlab.groot.data.GraphErrors
import org.jlab.groot.graphics.EmbeddedCanvasTabbed
import org.jlab.groot.base.GStyle

import org.jlab.jnp.utils.options.OptionStore
import org.jlab.jnp.utils.benchmark.ProgressPrintout

// --------------------------- Key & utils ---------------------------
class Key implements Comparable<Key>{
  final int s,l,c
  Key(int s,int l,int c){ this.s=s; this.l=l; this.c=c }
  int hashCode(){ Objects.hash(s,l,c) }
  boolean equals(Object o){ if(!(o instanceof Key)) return false; def k=(Key)o; return s==k.s && l==k.l && c==k.c }
  int compareTo(Key k){ int a=s-k.s; if(a!=0) return a; a=l-k.l; if(a!=0) return a; return c-k.c }
  String toString(){ "S${s} L${l} C${c}" }
}

double clamp(double x,double lo,double hi){ Math.max(lo, Math.min(hi,x)) }

// Histogram helpers (0-based bins)
double hIntegral(H1F h){
  double s=0; int nb=h.getAxis().getNBins()
  for(int i=0;i<nb;i++) s += h.getBinContent(i)
  return s
}
double quantile(H1F h,double q){
  q = clamp(q,0,1)
  int nb=h.getAxis().getNBins()
  double tot=0
  for(int i=0;i<nb;i++) tot += Math.max(0,h.getBinContent(i))
  if(tot<=0) return h.getAxis().min()
  double target=q*tot, cum=0
  for(int i=0;i<nb;i++){
    cum += Math.max(0,h.getBinContent(i))
    if(cum>=target) return h.getAxis().getBinCenter(i)
  }
  return h.getAxis().getBinCenter(nb-1)
}
double sigmaFromIQR(H1F h){
  double q1=quantile(h,0.25), q3=quantile(h,0.75)
  return 0.741*(q3-q1)
}
double modeQuad(H1F h){
  int nb=h.getAxis().getNBins()
  int im=0; double ymax=-1
  for(int i=0;i<nb;i++){ double y=h.getBinContent(i); if(y>ymax){ ymax=y; im=i } }
  if(im<=0 || im>=nb-1) return h.getAxis().getBinCenter(im)
  double y1=h.getBinContent(im-1), y2=h.getBinContent(im), y3=h.getBinContent(im+1)
  double x1=h.getAxis().getBinCenter(im-1), x2=h.getAxis().getBinCenter(im), x3=h.getAxis().getBinCenter(im+1)
  double denom=(x1-x2)*(x1-x3)*(x2-x3)
  if(Math.abs(denom)<1e-12) return x2
  double A=(x3*(y2-y1)+x2*(y1-y3)+x1*(y3-y2))/denom
  double B=(x3*x3*(y1-y2)+x2*x2*(y3-y1)+x1*x1*(y2-y3))/denom
  double xv = -B/(2*A)
  return clamp(xv, x1, x3)
}

// α,β fit on a control ADC window (Huber-weighted LS)
class AlphaBeta { final double a,b; AlphaBeta(double a,double b){ this.a=a; this.b=b } }
AlphaBeta fitAlphaBeta(H1F sig, H1F bkg, int iLo, int iHi){
  int nb=sig.getAxis().getNBins()
  iLo = clamp(iLo, 0, nb-1) as int
  iHi = clamp(iHi, 0, nb-1) as int
  if(iHi - iLo < 4) return new AlphaBeta(1.0, 0.0)
  int n=iHi-iLo+1
  double[] x=new double[n], y=new double[n], w=new double[n]
  int k=0
  for(int i=iLo;i<=iHi;i++){
    x[k]=Math.max(0,bkg.getBinContent(i))
    y[k]=Math.max(0,sig.getBinContent(i))
    w[k]=1.0/Math.max(1.0, x[k]+y[k])
    k++
  }
  double a=1.0, b=0.0
  for(int it=0; it<10; it++){
    double Sxx=0,Sxy=0,Sx=0,Sy=0,Sw=0
    for(int i=0;i<n;i++){
      double wi=w[i]; Sxx+=wi*x[i]*x[i]; Sxy+=wi*x[i]*y[i]; Sx+=wi*x[i]; Sy+=wi*y[i]; Sw+=wi
    }
    double det=Sxx*Sw - Sx*Sx
    if(Math.abs(det)<1e-12) break
    a=(Sxy*Sw - Sx*Sy)/det
    b=(Sxx*Sy - Sx*Sxy)/det
    // Huber reweight based on residuals
    double[] r=new double[n]
    for(int i=0;i<n;i++) r[i]=y[i]-(a*x[i]+b)
    double[] absr=r.collect{ Math.abs(it) } as double[]; Arrays.sort(absr)
    double mad = absr[(int)(0.5*n)]
    double scale=Math.max(1.0,1.4826*mad), c=1.345*scale
    for(int i=0;i<n;i++){
      double ri=Math.abs(r[i]); double hi=(ri<=c)?1.0:(c/ri)
      w[i]=hi / Math.max(1.0, x[i]+y[i])
    }
  }
  if(!Double.isFinite(a)) a=1.0
  if(!Double.isFinite(b)) b=0.0
  if(a<0) a=0
  return new AlphaBeta(a,b)
}

// MPV estimate + error from SUB
class MPVres { final double mpv, err; final int n; MPVres(double m,double e,int n){mpv=m;err=e;this.n=n} }
MPVres mpvAndErr(H1F h){
  int nb=h.getAxis().getNBins()
  double tot=0; for(int i=0;i<nb;i++) tot += Math.max(0,h.getBinContent(i))
  if(tot<20) return new MPVres(Double.NaN, Double.NaN, (int)tot)
  double mpv=modeQuad(h)
  // Width from IQR neighborhood around mode
  double bw=(h.getAxis().getBinCenter(1)-h.getAxis().getBinCenter(0))
  double sIQR=Math.max(bw, sigmaFromIQR(h))
  // Count in ±1.5σ window for error ~ bw/sqrt(Neff)
  int i0 = Math.max(0, (int)Math.floor((mpv-1.5*sIQR - h.getAxis().min())/bw))
  int i1 = Math.min(nb-1, (int)Math.floor((mpv+1.5*sIQR - h.getAxis().min())/bw))
  double Neff=0
  for(int i=i0;i<=i1;i++) Neff += Math.max(0,h.getBinContent(i))
  double err = bw/Math.sqrt(Math.max(Neff, 1))
  return new MPVres(mpv, err, (int)tot)
}

// Trimmed-median REF + error (MAD / sqrt(n))
class RefRes { final double ref, err; final int n; RefRes(double r,double e,int n){ref=r;err=e;this.n=n} }
RefRes refFromMPVs(List<Double> vals){
  List<Double> v=new ArrayList<>()
  for(double x: vals) if(Double.isFinite(x)) v.add(x)
  if(v.size()==0) return new RefRes(Double.NaN, Double.NaN, 0)
  Collections.sort(v)
  // IQR trimming (10% each side)
  int n=v.size()
  int lo=(int)Math.floor(0.10*n), hi=(int)Math.ceil(0.90*n)-1
  lo=clamp(lo,0,n-1) as int; hi=clamp(hi,0,n-1) as int
  List<Double> w=v.subList(lo, hi+1)
  double med = (w.size()%2==1) ? w[w.size()/2] : 0.5*(w[w.size()/2-1]+w[w.size()/2])
  // MAD
  double[] dev = new double[w.size()]
  for(int i=0;i<w.size();i++) dev[i]=Math.abs(w[i]-med)
  Arrays.sort(dev)
  double mad = dev[(int)(0.5*dev.length)]
  double err = (1.4826*mad)/Math.sqrt(Math.max(w.size(),1))
  return new RefRes(med, err, w.size())
}

// --------------------------- Config with CLI ---------------------------
@Field double PTMIN=0.24, PTMAX=0.29
@Field double K_SIG=1.1
@Field double SIG_FLOOR=100
@Field double VETO_NSIG=2.0
@Field double SB_DELTA=240
@Field double SB_WIDTH=140
@Field double ADC_CTRL_LO=0
@Field double ADC_CTRL_HI=300
@Field int    MAXEVENT=Integer.MAX_VALUE
@Field int    NB_ADC=90,  ADC_MIN=0,    ADC_MAX=4500
@Field int    NB_SUM=180, SUM_MIN=0,    SUM_MAX=4500
@Field int    NB_PT=50,   PT_MIN_H2=0,  PT_MAX_H2=1
@Field int    MIN_SUB_ENTRIES=25        // require for MPV measurement

void configureFromCLI(String[] args, OptionStore opts){
  PTMIN = opts.getOption("-ptmin")  ? Double.parseDouble(opts.getOption("-ptmin"))  : PTMIN
  PTMAX = opts.getOption("-ptmax")  ? Double.parseDouble(opts.getOption("-ptmax"))  : PTMAX
  K_SIG = opts.getOption("-k")      ? Double.parseDouble(opts.getOption("-k"))      : K_SIG
  SIG_FLOOR = opts.getOption("-sigFloor")? Double.parseDouble(opts.getOption("-sigFloor")): SIG_FLOOR
  VETO_NSIG  = opts.getOption("-vetoNSig")? Double.parseDouble(opts.getOption("-vetoNSig")): VETO_NSIG
  SB_DELTA   = opts.getOption("-sbDelta") ? Double.parseDouble(opts.getOption("-sbDelta")) : SB_DELTA
  SB_WIDTH   = opts.getOption("-sbWidth") ? Double.parseDouble(opts.getOption("-sbWidth")) : SB_WIDTH
  ADC_CTRL_LO= opts.getOption("-adcCtrlLo")? Double.parseDouble(opts.getOption("-adcCtrlLo")): ADC_CTRL_LO
  ADC_CTRL_HI= opts.getOption("-adcCtrlHi")? Double.parseDouble(opts.getOption("-adcCtrlHi")): ADC_CTRL_HI
  MAXEVENT   = opts.getOption("-nevent")? Integer.parseInt(opts.getOption("-nevent")): MAXEVENT
}

// --------------------------- Bank readers ---------------------------
// pT from AHDC::kftrack (MeV) or REC::Particle (GeV)
double eventPT(DataEvent ev){
  if(ev.hasBank("AHDC::kftrack")){
    DataBank t=ev.getBank("AHDC::kftrack")
    if(t.rows()>0){
      double px=t.getFloat("px",0), py=t.getFloat("py",0)
      return Math.sqrt(px*px+py*py)/1000.0
    }
  }
  if(ev.hasBank("REC::Particle")){
    DataBank p=ev.getBank("REC::Particle")
    if(p.rows()>0){
      double px=p.getFloat("px",0), py=p.getFloat("py",0)
      return Math.sqrt(px*px+py*py)
    }
  }
  return Double.NaN
}
double eventSumADC(DataEvent ev){
  if(!ev.hasBank("AHDC::adc")) return Double.NaN
  DataBank b=ev.getBank("AHDC::adc")
  boolean hasA=b.getDescriptor().hasEntry("ADC") || b.getDescriptor().hasEntry("adc")
  if(!hasA) return Double.NaN
  double s=0; int n=b.rows()
  for(int i=0;i<n;i++){
    int v=b.getDescriptor().hasEntry("ADC")? b.getInt("ADC",i): b.getInt("adc",i)
    s += Math.max(0,v)
  }
  return s
}
void forEachWireHit(DataEvent ev, Closure cb){
  if(!ev.hasBank("AHDC::adc")) return
  DataBank b=ev.getBank("AHDC::adc")
  boolean hasS=b.getDescriptor().hasEntry("sector")
  boolean hasL=b.getDescriptor().hasEntry("layer")
  boolean hasC=b.getDescriptor().hasEntry("component")
  boolean hasA=b.getDescriptor().hasEntry("ADC") || b.getDescriptor().hasEntry("adc")
  if(!(hasS&&hasL&&hasC&&hasA)) return
  for(int i=0;i<b.rows();i++){
    int s=b.getInt("sector",i)
    int l=b.getInt("layer",i)
    int c=b.getInt("component",i)
    int a=b.getDescriptor().hasEntry("ADC")? b.getInt("ADC",i): b.getInt("adc",i)
    cb.call(new Key(s,l,c), a)
  }
}

// --------------------------- Runner ---------------------------
class Runner {
  void go(String[] args){
    // Option parsing mirrors your run style (first token 'process')
    OptionStore opt = new OptionStore("process:")
    opt.addOption("-trackid", "1")       // accepted but unused here
    opt.addOption("-mode_valid","banana")// accepted but unused here
    opt.addOption("-ptmin","0.24")
    opt.addOption("-ptmax","0.29")
    opt.addOption("-k","1.1")
    opt.addOption("-sigFloor","100")
    opt.addOption("-vetoNSig","2.0")
    opt.addOption("-sbDelta","240")
    opt.addOption("-sbWidth","140")
    opt.addOption("-adcCtrlLo","0")
    opt.addOption("-adcCtrlHi","300")
    opt.addOption("-nevent",""+Integer.MAX_VALUE)
    opt.parse(args)

    configureFromCLI(args, opt)

    List<String> files = opt.getInputList()
    if(files==null || files.isEmpty()){
      System.err.println("[ERROR] No input files provided.")
      return
    }

    // Style
    GStyle.getAxisAttributesX().setTitleFontSize(18)
    GStyle.getAxisAttributesY().setTitleFontSize(18)
    GStyle.getAxisAttributesX().setLabelFontSize(14)
    GStyle.getAxisAttributesY().setLabelFontSize(14)
    GStyle.getH1FAttributes().setLineWidth(2)

    H2F h2_pt_sum = new H2F("pt_sum","p_{T} vs #Sigma ADC", NB_PT, PT_MIN_H2, PT_MAX_H2, NB_SUM, SUM_MIN, SUM_MAX)
    H1F h_sum_pt  = new H1F("sum_pt","Σ(ADC)", NB_SUM, SUM_MIN, SUM_MAX)

    Map<Key,H1F> hSIG=new TreeMap<>(), hBKG=new TreeMap<>(), hSUB=new TreeMap<>()

    def src = new HipoDataSource()
    ProgressPrintout prog = new ProgressPrintout()
    int nev=0

    // ---------- PASS 1: build ΣADC & initialize per-wire containers ----------
    for(String f: files){
      src.open(f)
      while(src.hasEvent() && nev<MAXEVENT){
        DataEvent ev=src.getNextEvent(); nev++; prog.updateStatus()
        double pt=eventPT(ev)
        double sum=eventSumADC(ev)
        if(Double.isFinite(pt) && Double.isFinite(sum)){
          h2_pt_sum.fill(pt,sum)
          if(pt>=PTMIN && pt<=PTMAX) h_sum_pt.fill(sum)
        }
        // ensure keys
        forEachWireHit(ev){ Key k,int adc->
          if(!hSIG.containsKey(k)){
            hSIG[k]=new H1F("SIG_"+k, "ADC — "+k+" (p_{T} slice) (SIG)", NB_ADC, ADC_MIN, ADC_MAX)
            hBKG[k]=new H1F("BKG_"+k, "ADC — "+k+" (p_{T} slice) (BKG)", NB_ADC, ADC_MIN, ADC_MAX)
          }
        }
      }
      src.close()
    }

    // ---------- Ridge finding within pT window ----------
    double muD = modeQuad(h_sum_pt)
    double sigD= Math.max(SIG_FLOOR, sigmaFromIQR(h_sum_pt))

    // Proton ridge from low region of ΣADC
    H1F h_low = new H1F("sum_low","Σ(ADC) low", NB_SUM, SUM_MIN, muD-2*sigD)
    int nb=h_sum_pt.getAxis().getNBins()
    for(int i=0;i<nb;i++){
      double x=h_sum_pt.getAxis().getBinCenter(i)
      if(x<=muD-2*sigD) h_low.setBinContent(i, h_sum_pt.getBinContent(i))
    }
    double muP = modeQuad(h_low)
    double sigP= Math.max(50.0, sigmaFromIQR(h_low))

    System.out.printf(Locale.US,
      "[GATES] pT=[%.3f,%.3f]  μD=%.1f σD=%.1f   μP=%.1f σP=%.1f%n",
      PTMIN,PTMAX,muD,sigD,muP,sigP)

    // ---------- PASS 2: classify events (SIG/BKG) ----------
    nev=0
    for(String f: files){
      src.open(f)
      while(src.hasEvent() && nev<MAXEVENT){
        DataEvent ev=src.getNextEvent(); nev++; prog.updateStatus()
        double pt=eventPT(ev); if(!Double.isFinite(pt) || pt<PTMIN || pt>PTMAX) continue
        double sum=eventSumADC(ev); if(!Double.isFinite(sum)) continue

        boolean vetoProton = sum > (muP + VETO_NSIG*sigP)
        boolean inSIG = Math.abs(sum - muD) < K_SIG*sigD && vetoProton
        boolean inSBm = (sum > (muD - SB_DELTA - SB_WIDTH)) && (sum < (muD - SB_DELTA))
        boolean inSBp = (sum > (muD + SB_DELTA)) && (sum < (muD + SB_DELTA + SB_WIDTH))
        boolean inBKG = (inSBm || inSBp) && vetoProton

        if(!(inSIG || inBKG)) continue
        forEachWireHit(ev){ Key k,int adc->
          if(inSIG) hSIG[k].fill(adc)
          else if(inBKG) hBKG[k].fill(adc)
        }
      }
      src.close()
    }

    // ---------- SUB = SIG - α*BKG - β per wire ----------
    int nbADC=NB_ADC
    double bw=(ADC_MAX-ADC_MIN)/ (double)NB_ADC
    int iCtrlLo = (int)Math.floor((ADC_CTRL_LO-ADC_MIN)/bw)
    int iCtrlHi = (int)Math.floor((ADC_CTRL_HI-ADC_MIN)/bw)
    iCtrlLo=Math.max(0,Math.min(nbADC-1,iCtrlLo))
    iCtrlHi=Math.max(iCtrlLo+4,Math.min(nbADC-1,iCtrlHi))

    Map<Key,AlphaBeta> abMap=new TreeMap<>()
    for(Key k: hSIG.keySet()){
      H1F s=hSIG[k], b=hBKG[k]
      s.setLineColor(1); b.setLineColor(2)
      AlphaBeta ab=fitAlphaBeta(s,b,iCtrlLo,iCtrlHi)
      abMap[k]=ab
      H1F u=new H1F("SUB_"+k, "ADC — "+k+" (p_{T} slice) (SUB=SIG-αBKG-β)", NB_ADC, ADC_MIN, ADC_MAX)
      for(int i=0;i<nbADC;i++){
        double y=s.getBinContent(i) - ab.a*b.getBinContent(i) - ab.b
        u.setBinContent(i, y)
      }
      u.setLineColor(4)
      hSUB[k]=u
    }

    // ---------- Per-wire MPV (from SUB) & Gain ----------
    List<Key> keys = new ArrayList<>(hSUB.keySet()); Collections.sort(keys)
    List<Double> mpvVals = new ArrayList<>()
    Map<Key,MPVres> mpvMap = new TreeMap<>()

    for(Key k: keys){
      H1F u=hSUB[k]
      int entries = (int)Math.round(hIntegral(u))
      if(entries >= MIN_SUB_ENTRIES){
        MPVres r = mpvAndErr(u)
        if(Double.isFinite(r.mpv) && Double.isFinite(r.err)){
          mpvMap[k]=r; mpvVals.add(r.mpv)
        }
      }
    }

    RefRes ref = refFromMPVs(mpvVals)
    System.out.printf(Locale.US,"[REF] MPV_ref=%.2f ± %.2f  (n=%d)%n", ref.ref, ref.err, ref.n)

    GraphErrors gMPV = new GraphErrors("MPV per wire")
    GraphErrors gGain= new GraphErrors("Gain per wire")
    gMPV.setTitle("MPV (SUB) vs wire"); gMPV.setMarkerSize(4)
    gGain.setTitle("Gain vs wire (REF/MPV)"); gGain.setMarkerSize(4)

    int idx=0
    for(Key k: keys){
      MPVres r = mpvMap.get(k)
      if(r==null) continue
      double mpv=r.mpv, empv=r.err
      gMPV.addPoint(idx, mpv, 0, empv)

      double gain = ref.ref / mpv
      double egain = Math.sqrt(
        (ref.err*ref.err)/(mpv*mpv) + (ref.ref*ref.ref*empv*empv)/(mpv*mpv*mpv*mpv)
      )
      gGain.addPoint(idx, gain, 0, egain)
      idx++
    }

    // ---------- Draw tabs ----------
    EmbeddedCanvasTabbed tabs = new EmbeddedCanvasTabbed("ΣADC","SIG","BKG","SUB","MPV","GAIN")
    // ΣADC
    tabs.getCanvas("ΣADC").draw(h_sum_pt)

    // grid drawer (15 per page)
    def drawGrid = { String tab, Collection<H1F> list, int color ->
      def cv = tabs.getCanvas(tab)
      cv.divide(5,3)
      int pad=0, per=15
      for(H1F h: list){
        if(pad>=per){ cv.update(); pad=0; cv.divide(5,3) }
        cv.cd(pad++)
        h.setLineColor(color)
        cv.draw(h)
      }
      cv.update()
    }
    drawGrid("SIG", hSIG.values(), 1)
    drawGrid("BKG", hBKG.values(), 2)
    drawGrid("SUB", hSUB.values(), 4)

    // MPV & GAIN graphs
    def cvM = tabs.getCanvas("MPV"); cvM.draw(gMPV)
    def cvG = tabs.getCanvas("GAIN"); cvG.draw(gGain)
    // thin baseline at y=1 for gains
    GraphErrors one = new GraphErrors("one"); one.addPoint(0,1,0,0); one.addPoint(Math.max(idx-1,1),1,0,0)
    one.setLineColor(1); one.setLineWidth(1); cvG.draw(one,"same")

    JFrame frame = new JFrame(String.format("AHDC Gains (pT=%.3f..%.3f)", PTMIN, PTMAX))
    frame.setSize(1500,950)
    frame.add(tabs)
    frame.setVisible(true)

    // Quick summary
    int ok=mpvMap.size(), tot=hSUB.size()
    System.out.printf(Locale.US,"[DONE] MPV measured on %d/%d wires (min SUB entries=%d)%n",
      ok, tot, MIN_SUB_ENTRIES)
  }
}

// --------------------------- Main entry ---------------------------
static void main(String[] args){
  if(args==null || args.length==0){
    System.err.println "Usage: run-groovy gain_MPV_proton_deutron_final_test.groovy process [options] files.hipo ..."
    System.err.println "Options: -trackid -mode_valid -ptmin -ptmax -k -sigFloor -vetoNSig -sbDelta -sbWidth -adcCtrlLo -adcCtrlHi -nevent"
    return
  }
  new Runner().go(args)
}
