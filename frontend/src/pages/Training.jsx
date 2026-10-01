import React, { useEffect, useState } from "react";
import api from "../api/client";
import { Notice, Page, Panel, Spinner, StatusBadge } from "../components/Common";
import { useApp } from "../context/AppContext";

const algorithms = ["LinearRegression", "Ridge", "RandomForest", "ExtraTrees", "GradientBoosting", "MLP", "XGBoost", "LightGBM", "CatBoost"];

export default function Training() {
  const { health, refresh } = useApp();
  const [form, setForm] = useState({test_size:.2, cv_folds:5, mode:"quick", models:algorithms.filter(name=>name!=="LightGBM")});
  const [result,setResult]=useState(null); const [status,setStatus]=useState(null); const [busy,setBusy]=useState(false); const [error,setError]=useState("");
  useEffect(()=>{api.get("/training/results").then(r=>setResult(r.data)).catch(()=>{});},[]);
  const toggle=name=>setForm({...form,models:form.models.includes(name)?form.models.filter(item=>item!==name):[...form.models,name]});
  const train=async()=>{setBusy(true);setError("");setStatus({state:"preprocessing",message:"Starting training"});const timer=setInterval(()=>api.get("/training/status").then(r=>setStatus(r.data)).catch(()=>{}),1000);try{const data=(await api.post("/training/train",form)).data;setResult(data);await refresh();}catch(e){setError(e.message);}finally{clearInterval(timer);setBusy(false)}};
  return <Page title="Model Training" subtitle="Compare reproducible regression pipelines and activate the best validated model.">
    {!health?.schema_configured&&<Notice type="warning">Configure dataset columns before training.</Notice>}{error&&<Notice type="error">{error}</Notice>}
    <Panel title="Training controls"><div className="form-grid"><label className="field"><span>Test fraction</span><input type="number" min="0.1" max="0.4" step="0.05" value={form.test_size} onChange={e=>setForm({...form,test_size:+e.target.value})}/></label><label className="field"><span>Cross-validation folds</span><input type="number" min="2" max="10" value={form.cv_folds} onChange={e=>setForm({...form,cv_folds:+e.target.value})}/></label><label className="field"><span>Tuning mode</span><select value={form.mode} onChange={e=>setForm({...form,mode:e.target.value})}><option value="quick">Quick</option><option value="standard">Standard randomized search</option></select></label></div><div className="algorithm-list">{algorithms.map(name=><label key={name}><input type="checkbox" checked={form.models.includes(name)} onChange={()=>toggle(name)}/>{name}</label>)}</div>{status&&busy&&<Notice>{status.state}: {status.message}</Notice>}<button className="primary" disabled={busy||!health?.schema_configured||form.models.length===0} onClick={train}>{busy?<Spinner label="Training models..."/>:"Train and compare models"}</button></Panel>
    {result&&<Panel title={`Selected model: ${result.active_model}`} subtitle="Validation metrics from the most recent reproducible training run"><div className="table-scroll"><table><thead><tr><th>Model</th><th>Status</th><th>Test R²</th><th>MAE</th><th>RMSE</th><th>CV R²</th><th>Time</th><th>Parameters</th></tr></thead><tbody>{Object.entries(result.metrics).map(([name,row])=>{const values=Object.values(row.metrics||{});const average=key=>values.length?values.reduce((sum,item)=>sum+item[key],0)/values.length:null;return <tr key={name}><td><strong>{name}</strong></td><td><StatusBadge tone={row.status==="trained"?"success":row.status==="unavailable"?"warning":"danger"}>{row.status}</StatusBadge></td><td>{average("r2")?.toFixed(4)??"--"}</td><td>{average("mae")?.toFixed(3)??"--"}</td><td>{average("rmse")?.toFixed(3)??"--"}</td><td>{row.cv_r2?.toFixed(4)??"--"}</td><td>{row.training_time?.toFixed(2)} s</td><td>{Object.keys(row.best_parameters||{}).length?JSON.stringify(row.best_parameters):row.status==="trained"?"Default":row.reason}</td></tr>})}</tbody></table></div></Panel>}
  </Page>;
}
