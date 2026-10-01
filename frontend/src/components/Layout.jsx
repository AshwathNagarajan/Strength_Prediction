import React, { useState } from "react";
import { Activity, BarChart3, Brain, CircleCheck, Database, FlaskConical, Gauge, Leaf, Menu, Server, SlidersHorizontal, X } from "lucide-react";
import { NavLink } from "react-router-dom";
import { useApp } from "../context/AppContext";
const links = [["/", "Dashboard", Activity], ["/dataset", "Dataset", Database], ["/training", "Model Training", FlaskConical], ["/prediction", "Prediction", Gauge], ["/explainability", "Explainability", Brain], ["/optimization", "Optimization", SlidersHorizontal], ["/results", "Results", BarChart3]];
export default function Layout({ children }) {
  const { health } = useApp();
  const [menuOpen,setMenuOpen]=useState(false);
  const closeMenu=()=>setMenuOpen(false);
  return <div className="app-shell"><a className="skip-link" href="#main-content">Skip to main content</a>
    <header className="mobile-header"><button className="icon-button" title="Open navigation" onClick={()=>setMenuOpen(true)}><Menu size={20}/></button><div><Leaf size={20}/><strong>Sustainable Beam AI</strong></div><span className={health?.model_trained?"mobile-status ready":"mobile-status"}/></header>
    {menuOpen&&<button className="nav-scrim" aria-label="Close navigation" onClick={closeMenu}/>} 
    <aside className={`nav-panel ${menuOpen?"open":""}`}><div className="brand"><span className="brand-mark"><Leaf size={25}/></span><div><strong>Sustainable Beam AI</strong><span>Engineering research platform</span></div><button className="nav-close" title="Close navigation" onClick={closeMenu}><X size={18}/></button></div><div className="nav-label">Workspace</div><nav>{links.map(([to,label,Icon]) => <NavLink key={to} to={to} end={to === "/"} onClick={closeMenu}><Icon size={18}/><span>{label}</span></NavLink>)}</nav><div className="system-state"><span className={health?.model_trained ? "dot ready" : "dot"}/><div><strong>{health?.model_trained ? "Model ready" : "Setup required"}</strong><span>{health?.dataset_configured?"Dataset connected":"Upload a dataset to begin"}</span></div></div></aside>
    <main className="workspace" id="main-content"><div className="workspace-inner"><div className="utility-bar"><div className="utility-context"><span className="environment-badge">Local research environment</span><span className="utility-divider"/><span className="utility-name">Composite beam programme</span></div><div className="utility-status"><span><Server size={14}/>API connected</span><span className={health?.model_trained?"healthy":"pending"}><CircleCheck size={14}/>{health?.model_trained?"Active model ready":"Model setup pending"}</span></div></div>{children}<footer><strong>Engineering use notice</strong><span>This system is a research and decision-support tool based on experimental data and machine-learning predictions. Final structural design must be verified using applicable design codes, engineering calculations, and qualified professional review.</span></footer></div></main>
  </div>;
}
