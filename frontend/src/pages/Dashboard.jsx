import React, { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import { ArrowRight, Database, FlaskConical, Gauge, SlidersHorizontal } from "lucide-react";
import { Link } from "react-router-dom";
import api from "../api/client";
import { Metric, Notice, Page, Panel } from "../components/Common";
import { useApp } from "../context/AppContext";

export default function Dashboard() {
  const { health, schema } = useApp(); const [model, setModel] = useState(null); const [info, setInfo] = useState(null); const [importance,setImportance]=useState([]);
  useEffect(() => { api.get("/models/best").then(r=>setModel(r.data)).catch(()=>{}); api.get("/dataset/info").then(r=>setInfo(r.data)).catch(()=>{}); api.get("/reports/feature-importance?format=json").then(r=>setImportance(r.data.importance||[])).catch(()=>{}); }, []);
  const rows = model ? Object.entries(model.metrics).filter(([,v])=>v.status==="trained").map(([name,v])=>({name, cv:Number(v.cv_r2.toFixed(3))})) : [];
  const activeMetrics=model?.metrics?.[model?.active_model]?.metrics||{}; const targetMetrics=Object.values(activeMetrics); const average=key=>targetMetrics.length?targetMetrics.reduce((sum,row)=>sum+row[key],0)/targetMetrics.length:null;
  const workflow=[
    {label:"Dataset",detail:health?.dataset_configured?"Configured and validated":"Upload experimental data",ready:health?.dataset_configured,to:"/dataset",Icon:Database},
    {label:"Model training",detail:health?.model_trained?`${model?.active_model||"Model"} is active`:"Train and compare regressors",ready:health?.model_trained,to:"/training",Icon:FlaskConical},
    {label:"Prediction",detail:"Evaluate a beam configuration",ready:health?.model_trained,to:"/prediction",Icon:Gauge},
    {label:"Optimization",detail:"Search bounded design alternatives",ready:health?.model_trained,to:"/optimization",Icon:SlidersHorizontal},
  ];
  return <Page title="Engineering Dashboard" subtitle="Dataset, model, and experimental-domain readiness at a glance." actions={<><Link className="button-link" to="/prediction">New prediction</Link><Link className="button-link primary" to="/optimization">Run optimization</Link></>}>
    {!health?.dataset_configured && <Notice>No dataset configured. Open Dataset to begin.</Notice>}
    {health?.dataset_configured && !health?.model_trained && <Notice type="warning">Dataset configured. Train a model to enable prediction and optimization.</Notice>}
    <div className="section-label"><span>Operational summary</span><small>Live project state</small></div>
    <div className="metrics"><Metric label="Dataset rows" value={info?.rows}/><Metric label="Input features" value={schema?.columns?.filter(c=>c.role==="input").length}/><Metric label="Targets" value={model?.targets?.length}/><Metric label="Best model" value={model?.active_model}/><Metric label="Test R²" value={average("r2")?.toFixed(3)}/><Metric label="RMSE" value={average("rmse")?.toFixed(3)}/><Metric label="MAE" value={average("mae")?.toFixed(3)}/><Metric label="Training date" value={model?.trained_at ? new Date(model.trained_at).toLocaleDateString() : null}/><Metric label="Dataset domain" value={health?.dataset_configured?"Configured":"Unavailable"}/></div>
    <div className="workflow-strip">{workflow.map(({label,detail,ready,to,Icon})=><Link to={to} key={label} className="workflow-item"><span className="workflow-icon"><Icon size={18}/></span><span><strong>{label}</strong><small>{detail}</small></span><span className={ready?"workflow-state ready":"workflow-state"}>{ready?"Ready":"Action"}</span><ArrowRight size={16}/></Link>)}</div>
    <div className="section-label"><span>Model intelligence</span><small>Validation and readiness</small></div>
    <div className="split"><Panel title="Model comparison" subtitle="Cross-validation R² by trained algorithm"><div className="chart"><ResponsiveContainer><BarChart data={rows} margin={{top:10,right:10,left:-12,bottom:5}}><CartesianGrid vertical={false} stroke="#dce4e5"/><XAxis dataKey="name" tick={{fontSize:11}}/><YAxis tick={{fontSize:11}}/><Tooltip/><Bar dataKey="cv" fill="#15705c" radius={[3,3,0,0]}/></BarChart></ResponsiveContainer></div></Panel><Panel title="Workflow readiness" subtitle="Required controls for numerical analysis"><ul className="status-list"><li className={health?.dataset_configured?"done":""}>Dataset uploaded</li><li className={health?.schema_configured?"done":""}>Columns configured</li><li className={health?.model_trained?"done":""}>Model trained</li></ul></Panel></div>
    {model?.evaluation&&Object.entries(model.evaluation).map(([target,points])=><Panel key={target} title={`Actual vs predicted: ${target}`}><div className="chart"><ResponsiveContainer><ScatterChart><CartesianGrid/><XAxis dataKey="actual" name="Actual"/><YAxis dataKey="predicted" name="Predicted"/><Tooltip cursor={{strokeDasharray:"3 3"}}/><Scatter data={points} fill="#236a83"/></ScatterChart></ResponsiveContainer></div></Panel>)}
    {importance.length>0&&<Panel title="SHAP feature importance" subtitle="Mean absolute contribution across the active model targets"><div className="chart"><ResponsiveContainer><BarChart layout="vertical" data={importance.slice(0,10)}><CartesianGrid/><XAxis type="number"/><YAxis type="category" dataKey="feature" width={170}/><Tooltip/><Bar dataKey="mean_abs_shap" fill="#236a83"/></BarChart></ResponsiveContainer></div></Panel>}
  </Page>;
}
