import React, { useState } from "react";
import Plot from "react-plotly.js";
import { Download } from "lucide-react";
import { Notice, Page, Panel, StatusBadge } from "../components/Common";
import { useApp } from "../context/AppContext";

const base = import.meta.env.VITE_API_URL || "http://localhost:8000/api";
const openReport = path => window.open(base + path, "_blank", "noopener,noreferrer");

export default function Results() {
  const { prediction, optimization } = useApp();
  const [selectedPareto,setSelectedPareto]=useState(0);
  const pareto = optimization?.pareto_front || [];
  const objectives = pareto.map(row => row.objectives);
  return <Page title="Results" subtitle="Inspect and export the current prediction and optimization evidence.">
    {!prediction && !optimization && <Notice>No results yet. Run prediction or optimization first.</Notice>}
    {prediction && <Panel title="Prediction result">
      <button onClick={() => openReport("/reports/prediction?format=json")}><Download size={16}/>Prediction JSON</button>
      <div className="metrics">{Object.entries(prediction.predictions).map(([name,value]) => <div className="metric" key={name}><span>{name}</span><strong>{value.toFixed(3)}</strong></div>)}</div>
      <p>{prediction.natural_language_explanation}</p>
    </Panel>}
    {optimization && <Panel title="Optimization result">
      <p>{optimization.selection_rule}</p>
      <div className="actions">
        <button onClick={() => openReport("/reports/optimization?format=json")}><Download size={16}/>Result JSON</button>
        <button onClick={() => openReport("/reports/optimization?format=csv")}><Download size={16}/>Solutions CSV</button>
        <button onClick={() => openReport("/reports/model-comparison?format=csv")}><Download size={16}/>Model metrics CSV</button>
        <button onClick={() => openReport("/reports/feature-importance?format=csv")}><Download size={16}/>SHAP CSV</button>
      </div>
      {optimization.best_solution && <div className="table-scroll"><table><tbody>{Object.entries(optimization.best_solution.features).map(([name,value]) => <tr key={name}><th>{name}</th><td>{String(value)}</td></tr>)}</tbody></table></div>}
      {optimization.best_solution && <><h3>Target comparison</h3><div className="table-scroll"><table><thead><tr><th>Target</th><th>Constraint</th><th>Required</th><th>Predicted</th><th>Status</th></tr></thead><tbody>{optimization.best_solution.constraints.map(item=><tr key={item.target}><td>{item.target}</td><td>{item.operator}</td><td>{item.required}</td><td>{Number(item.predicted).toFixed(3)}</td><td><StatusBadge tone={item.satisfied?"success":"danger"}>{item.satisfied?"Satisfied":"Violated"}</StatusBadge></td></tr>)}</tbody></table></div><p>Domain position: <StatusBadge tone={optimization.best_solution.domain_reliability?.includes("boundary")?"warning":"success"}>{optimization.best_solution.domain_reliability}</StatusBadge></p></>}
      {optimization.alternatives?.length>0&&<><h3>Alternative feasible designs</h3><div className="table-scroll"><table><thead><tr><th>Solution</th><th>Predictions</th><th>Target deviation</th><th>Domain similarity</th></tr></thead><tbody>{optimization.alternatives.map((row,index)=><tr key={index}><td>{index+2}</td><td>{Object.entries(row.predictions).map(([name,value])=>`${name}: ${Number(value).toFixed(3)}`).join("; ")}</td><td>{Number(row.target_deviation).toFixed(3)}</td><td>{row.domain_reliability}</td></tr>)}</tbody></table></div></>}
      {objectives.length > 0 && <><Plot data={[{x:objectives.map(v=>v[0]),y:objectives.map(v=>v[1]??0),mode:"markers",type:"scatter",text:pareto.map((_,i)=>`Solution ${i+1}`),marker:{color:pareto.map((_,i)=>i===selectedPareto?"#c2473d":"#1f7a5a"),size:10}}]} layout={{title:"Pareto front",xaxis:{title:"Objective 1"},yaxis:{title:"Objective 2"},autosize:true}} onClick={event=>setSelectedPareto(event.points[0].pointIndex)} useResizeHandler style={{width:"100%",height:420}}/><h3>Selected Pareto design</h3><pre>{JSON.stringify(pareto[selectedPareto],null,2)}</pre></>}
      <p>{optimization.natural_language_explanation}</p>
    </Panel>}
  </Page>;
}
