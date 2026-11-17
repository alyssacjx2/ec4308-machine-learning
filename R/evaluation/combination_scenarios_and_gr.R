# combination_scenarios_and_gr.R -------------------------------------------------
# Two equal-weight combinations (A: original candidates, B: adds the 168 h and holiday
# SARIMA/Holt-Winters variants), Granger-Ramanathan (GR) weights, and DM tests vs AR with Holm.
# SARIMA is the per-horizon fit. Members/weights use VALIDATION only; all scores are TEST.
# DM: squared error, Newey-West Bartlett, 20 lags (168 sensitivity), no prewhitening, Student-t df n-1;
# Holm within {individual models} and {combination} (A: 36+4, B: 60+4); GR methods: Holm over 9 x 4.
# Run from the repo root:  Rscript R/evaluation/combination_scenarios_and_gr.R
# in : output/forecasts/{benchmarks,lasso,lgbm,rf_bagging}.csv.gz, sarima_holt_winters_forecast_results.zip, sarima_hw_forecasts.zip
# out: output/tables/{combination_members_A_B,validation_rmse_all_candidates,test_metrics_A_B,dm_tests_holm_A_B,
#      granger_ramanathan_metrics,granger_ramanathan_weights,granger_ramanathan_dm}.csv
suppressPackageStartupMessages(library(data.table))
R <- "."; FC <- file.path(R,"output/forecasts"); OUT <- file.path(R,"output/tables")
dir.create(OUT, recursive=TRUE, showWarnings=FALSE)
W <- tempfile(); dir.create(W)
unzip(file.path(FC,"sarima_hw_forecasts.zip"), exdir=W); unzip(file.path(FC,"sarima_holt_winters_forecast_results.zip"), exdir=W)
source(file.path(R,"R/utils.R"))
cc <- c(datetime="character")
bm <- read_gz(file.path(FC,"benchmarks.csv.gz"), colClasses=cc)
la <- read_gz(file.path(FC,"lasso.csv.gz"), colClasses=cc)
lg <- read_gz(file.path(FC,"lgbm.csv.gz"), colClasses=cc)
fo <- read_gz(file.path(FC,"rf_bagging.csv.gz"), colClasses=c(datetime="character",target_datetime="character"))
setnames(fo, c("horizon","actual_mw","rf_forecast_mw","bagging_forecast_mw"), c("h","actual","random_forest","bagging"))
sh <- fread(file.path(W,"sarima_holt_winters_forecast_results.csv"), colClasses=c(origin="character",target_datetime="character"))
setnames(sh,"origin","datetime")
nh <- fread(file.path(W,"forecasts_without_holiday.csv"), colClasses=c(datetime="character"))
wh <- fread(file.path(W,"forecasts_with_holiday.csv"), colClasses=c(datetime="character"))
setnames(nh, c("sarima24_direct","sarima168_direct","hw24","hw168"), c("sarima24","sarima168","hw24_b","hw168"))
setnames(wh, c("sarima24_direct_holiday","sarima168_direct_holiday","hw24_holiday","hw168_holiday"), c("sarima24_hol","sarima168_hol","hw24_hol","hw168_hol"))
d <- Reduce(function(a,b) merge(a,b,by=c("datetime","h")),
  list(bm[,.(datetime,h,actual,snaive168,snaive24,ar_bic,ema)], la[,.(datetime,h,lasso,post_lasso)], lg[,.(datetime,h,lgbm)],
       sh[,.(datetime,h,period,hw24)], fo[,.(datetime,h,random_forest,bagging)],
       nh[,.(datetime,h,sarima24,sarima168,hw168)], wh[,.(datetime,h,sarima24_hol,sarima168_hol,hw24_hol,hw168_hol)]))
cat("rows",nrow(d),"; periods:"); print(d[,.N,by=.(period)])
# check actuals consistent
a2 <- merge(bm[,.(datetime,h,a=actual)], fo[,.(datetime,h,b=actual)], by=c("datetime","h")); cat("max actual diff rf:",max(abs(a2$a-a2$b)),"\n")
base <- c("ar_bic","lasso","post_lasso","lgbm","random_forest","bagging","snaive168","snaive24")
A <- c(base,"sarima24","hw24")
B <- c(A,"sarima24_hol","sarima168","sarima168_hol","hw24_hol","hw168","hw168_hol")
val <- d[period=="validation"]; te <- d[period=="test"]
cat("val n/h:",val[h==1,.N]," test n/h:",te[h==1,.N],"\n")
rm_ <- function(x,m,hh) sqrt(mean((x[h==hh]$actual - x[h==hh][[m]])^2))
vr <- rbindlist(lapply(c(1,6,12,24), function(hh) data.table(h=hh, model=B, val_rmse=sapply(B,function(m) rm_(val,m,hh)))))
fwrite(vr, file.path(OUT,"validation_rmse_all_candidates.csv"))
print(dcast(vr, model~h, value.var="val_rmse")[order(`24`)])
mk <- function(cands, name){
  mem <- rbindlist(lapply(c(1,6,12,24), function(hh){ r <- vr[h==hh & model %in% cands]; r <- r[order(val_rmse)]; data.table(h=hh, scenario=name, members=paste(r$model[1:3],collapse="+")) }))
  col <- paste0("comb_",name); d[, (col):=NA_real_]
  for(hh in mem$h){ ms <- strsplit(mem[h==hh,members],"\\+")[[1]]; d[h==hh, (col):=rowMeans(.SD), .SDcols=ms] }
  mem }
memA <- mk(A,"A"); memB <- mk(B,"B"); mem <- rbind(memA,memB); print(mem); fwrite(mem, file.path(OUT,"combination_members_A_B.csv"))
te <- d[period=="test"]
met <- rbindlist(lapply(c(1,6,12,24), function(hh) rbindlist(lapply(c(B,"comb_A","comb_B","ema"), function(m){
  x<-te[h==hh]; e<-x$actual-x[[m]]; data.table(h=hh,model=m,n=length(e),rmse=sqrt(mean(e^2)),mae=mean(abs(e)),mape=100*mean(abs(e)/x$actual),bias=mean(e)) }))))
fwrite(met, file.path(OUT,"test_metrics_A_B.csv"))
print(dcast(met,model~h,value.var="rmse")[order(`24`)])
dm_test <- function(l1,l2,lag=20L){ x<-l1-l2; x<-x[is.finite(x)]; n<-length(x); xb<-mean(x); xc<-x-xb
  g<-function(k) sum(xc[(k+1):n]*xc[1:(n-k)])/n
  lrv<-g(0)+2*sum(vapply(seq_len(lag),function(k)(1-k/(lag+1))*g(k),0)); s<-xb/sqrt(lrv/n); c(stat=s,p=2*pt(-abs(s),df=n-1)) }
run_dm <- function(models, comb, lag, scen){
  dm <- rbindlist(lapply(c(1,6,12,24), function(hh){ x<-te[h==hh]; eref<-x$actual-x$ar_bic
    rbindlist(lapply(c(setdiff(models,"ar_bic"),comb), function(m){ e<-x$actual-x[[m]]; se<-dm_test(e^2,eref^2,lag); ae<-dm_test(abs(e),abs(eref),lag)
      data.table(scenario=scen,lags=lag,h=hh,model=m,DM_se=se["stat"],p_se=se["p"],DM_ae=ae["stat"],p_ae=ae["p"]) })) }))
  dm[, grp:=fifelse(model==comb,"combination","individual")]
  dm[, p_se_holm:=p.adjust(p_se,"holm"), by=grp]; dm[, family_size:=.N, by=grp]
  dm[, concl:=fifelse(p_se_holm>=0.05,"no_detectable_difference",fifelse(DM_se<0,"model_better","ar_better"))]
  dm[, model:=fifelse(model==comb,"combination",model)]; dm }
dms <- rbind(run_dm(A,"comb_A",20L,"A"), run_dm(B,"comb_B",20L,"B"), run_dm(A,"comb_A",168L,"A"), run_dm(B,"comb_B",168L,"B"))
fwrite(dms, file.path(OUT,"dm_tests_holm_A_B.csv"))


# ---- Granger-Ramanathan ----

base <- c("ar_bic","lasso","post_lasso","lgbm","random_forest","bagging","snaive168","snaive24")
A <- c(base,"sarima24","hw24")
B <- c(A,"sarima24_hol","sarima168","sarima168_hol","hw24_hol","hw168","hw168_hol")
val <- d[period=="validation"]; te <- d[period=="test"]
# GR_noint: unrestricted OLS weights, no intercept.
# GR_int: OLS weights plus an intercept.
# GR_sum1: subtract the last forecast so the recovered weights sum to one.
# Singular OLS coefficients are replaced with zero by the existing convention.
fit_gr <- function(y, X, type){
  if(type=="GR_noint"){ b <- coef(lm(y ~ X - 1)); w <- b; ic <- 0 }
  if(type=="GR_int"){ b <- coef(lm(y ~ X)); ic <- b[1]; w <- b[-1] }
  if(type=="GR_sum1"){ k <- ncol(X); Z <- X[,-k,drop=FALSE]-X[,k]; b <- coef(lm((y-X[,k]) ~ Z - 1)); w <- c(b, 1-sum(b)); ic <- 0 }
  w[is.na(w)] <- 0; list(w=as.numeric(w), ic=as.numeric(ic)) }
sets <- list(top3=NULL, A=A, B=B)
types <- c("GR_noint","GR_sum1","GR_int")
res <- list(); Wt <- list()
for(hh in c(1,6,12,24)){
  v <- val[h==hh]; t <- te[h==hh]
  for(s in names(sets)) for(ty in types){
    ms <- if(s=="top3") strsplit(mem[h==hh & scenario=="A", members],"\\+")[[1]] else sets[[s]]
    f <- fit_gr(v$actual, as.matrix(v[, ..ms]), ty)
    pred <- as.numeric(as.matrix(t[, ..ms]) %*% f$w) + f$ic
    vp <- as.numeric(as.matrix(v[, ..ms]) %*% f$w) + f$ic
    nm <- paste(ty, s, sep=".")
    res[[length(res)+1]] <- data.table(h=hh, method=nm, val_rmse=sqrt(mean((v$actual-vp)^2)), test_rmse=sqrt(mean((t$actual-pred)^2)),
        test_mae=mean(abs(t$actual-pred)), test_mape=100*mean(abs(t$actual-pred)/t$actual), sum_w=sum(f$w), intercept=f$ic)
    Wt[[length(Wt)+1]] <- data.table(h=hh, method=nm, model=ms, weight=f$w)
    d[h==hh & period=="test", (nm):=pred]
  }
}
res <- rbindlist(res); Wt <- rbindlist(Wt)
cmp <- rbindlist(lapply(c(1,6,12,24), function(hh) { x<-te[h==hh]; data.table(h=hh, method=c("ar_bic","comb_equal"), val_rmse=NA, test_rmse=c(sqrt(mean((x$actual-x$ar_bic)^2)), sqrt(mean((x$actual-x$comb_A)^2))))}), fill=TRUE)
options(width=220)
print(dcast(res, method~h, value.var="test_rmse")); print(dcast(res, method~h, value.var="val_rmse")); print(dcast(res,method~h,value.var="sum_w"))
print(Wt[method %in% c("GR_noint.top3","GR_sum1.top3")][order(method,h)])
print(Wt[method=="GR_sum1.B" & abs(weight)>0.05][order(h,-abs(weight))])
fwrite(res,file.path(OUT,"granger_ramanathan_metrics.csv")); fwrite(Wt,file.path(OUT,"granger_ramanathan_weights.csv"))


# ---- DM for GR ----
te <- d[period=="test"]
dm_test <- function(l1,l2,lag=20L){ z<-l1-l2; n<-length(z); zb<-mean(z); zc<-z-zb; g<-function(k) sum(zc[(k+1):n]*zc[1:(n-k)])/n
  lrv<-g(0)+2*sum(vapply(seq_len(lag),function(k)(1-k/(lag+1))*g(k),0)); s<-zb/sqrt(lrv/n); c(stat=s,p=2*pt(-abs(s),df=n-1)) }
ms <- grep("^GR_", names(te), value=TRUE)
out <- rbindlist(lapply(c(20L,168L), function(lag) rbindlist(lapply(c(1,6,12,24), function(hh){ t<-te[h==hh]; ea<-t$actual-t$ar_bic; ec<-t$actual-t$comb_A
  rbindlist(lapply(ms, function(m){ e<-t$actual-t[[m]]; a<-dm_test(e^2,ea^2,lag); b<-dm_test(e^2,ec^2,lag)
    data.table(lags=lag,h=hh,method=m,DM_vs_AR=a["stat"],p_vs_AR=a["p"],DM_vs_comb=b["stat"],p_vs_comb=b["p"]) })) }))))
out[, `:=`(holm_vs_AR=p.adjust(p_vs_AR,"holm"), holm_vs_comb=p.adjust(p_vs_comb,"holm")), by=lags]
fwrite(out,file.path(OUT,"granger_ramanathan_dm.csv"))
f<-function(p) ifelse(p<0.001,"<0.001",sprintf("%.3f",p)); options(width=200)
for(lg in c(20L,168L)){ cat("\nlags",lg,"\n"); y<-out[lags==lg]
 print(dcast(y,method~h,value.var="holm_vs_AR")[,lapply(.SD,function(z) if(is.numeric(z)) f(z) else z)])
 print(dcast(y,method~h,value.var="DM_vs_AR")[,lapply(.SD,function(z) if(is.numeric(z)) round(z,2) else z)])
 print(dcast(y,method~h,value.var="holm_vs_comb")[,lapply(.SD,function(z) if(is.numeric(z)) f(z) else z)])
 print(dcast(y,method~h,value.var="DM_vs_comb")[,lapply(.SD,function(z) if(is.numeric(z)) round(z,2) else z)]) }
# bias check on h=24 weights negative; metrics MAE vs combo
print(rbindlist(lapply(c(1,6,12,24),function(hh){t<-te[h==hh]; data.table(h=hh,comb_mae=mean(abs(t$actual-t$comb_A)),grA_mae=mean(abs(t$actual-t$GR_noint.A)),grTop_mae=mean(abs(t$actual-t$GR_noint.top3)))})))
