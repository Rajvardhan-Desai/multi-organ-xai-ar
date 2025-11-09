"use client";
import React from "react";
type Item={name:string;score:number;};
export default function BarChart({items}:{items:Item[]}) {
  const max=Math.max(1e-6,...items.map(i=>Math.abs(i.score)));
  return(
    <div className="card" style={{width:"100%",maxWidth:520}}>
      <h3 style={{marginTop:0}}>Top ROI contributions</h3>
      <div style={{display:"flex",flexDirection:"column",gap:8}}>
        {items.map((i,idx)=>{
          const w=Math.round(Math.abs(i.score)/max*100);
          const sign=i.score>=0?"+":"−";
          return (
            <div key={idx}>
              <div style={{display:"flex",justifyContent:"space-between"}}>
                <span style={{color:"#cfe1ff"}}>{i.name}</span>
                <span style={{color:"#9fb3c8",fontVariantNumeric:"tabular-nums"}}>{sign}{Math.abs(i.score).toFixed(3)}</span>
              </div>
              <div style={{height:8,background:"#0c111a",borderRadius:6}}>
                <div style={{width:`${w}%`,height:"100%",borderRadius:6,background:"#66d9ef"}}/>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
