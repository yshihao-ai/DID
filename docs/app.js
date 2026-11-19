'use strict';
const $ = id => document.getElementById(id);
const C = {ink:'#262b3d',muted:'#72798b',line:'#dfe3ec',purple:'#7150d3',pl:'#eee8ff',teal:'#13877d',tl:'#e5f5f0',amber:'#a67420',al:'#fff4df'};
const score = [[2,4,3,6,5],[4,2,1],[1,3,5,4,6,3],[4,3,2,1]];
const phases = [];
score.forEach((notes,t)=>{
  phases.push({t,kind:'predict',n:0,duration:1900},{t,kind:'inject',n:0,duration:1500});
  notes.forEach((_,n)=>phases.push({t,kind:'event',n:n+1,duration:850}));
  phases.push({t,kind:'stop',n:notes.length,duration:1400},{t,kind:'summary',n:notes.length,duration:1400},{t,kind:'feedback',n:notes.length,duration:1800});
});
phases.push({t:3,kind:'done',n:score[3].length,duration:2200});
let index=0, lang='en', playing=false, looping=false, timer=null, speed=1;
const initialEnglish = Object.fromEntries([...document.querySelectorAll('[data-i18n]')].map(e=>[e.dataset.i18n,e.innerHTML]));
const zh = {
  brand:'Decoder-in-Decoder',interactive:'交互式原理动画',kicker:'二维自回归音乐生成',title:'音乐，沿着<em>两个维度展开。</em>',subtitle:'在一个小节内逐事件生成。完成这一节，才能进入下一节。',across:'跨小节 · 纵向推进',within:'小节内 · 横向展开',analogy:'打印机类比',printer:'一台打印乐谱的机器',sync:'同步演示',mechanism:'模型生成流程',decoder:'两个解码器，共同展开一张乐谱。',start:'已有历史',finish:'乐谱完成',speed:'速度',loop:'循环',takeaway:'<strong>关键依赖：</strong>下一小节必须等待当前小节生成结束。',schematic:'概念示意动画 · 乐谱仅作演示',details:'预测了什么？与实际代码如何对应？',detailTitle1:'隐藏状态，就是对下一小节的指导。',detailBody1:'在时间步 t，Time Decoder 根据可用历史产生面向第 t 小节的隐藏状态 hₜ。可学习的连接器将 hₜ 映射成 4 个隐状态前缀向量 Pₜ。此前缀被注入 Note Decoder，指导它自回归地生成第 t 小节的音乐事件。这些指导是隐空间表示，并不是已经生成的音符。',detailBody2:'“预测指导”指模型产生用于预测目标小节事件的隐空间表示，并不意味着代码另有一个针对未来隐藏状态的监督目标或独立损失函数。',detailTitle2:'有意保留的概念简化。',detailBody3:'主动画呈现概念循环：完成第 t 小节、得到总结、反馈给下一时间步。当前 main 分支把历史分成两路：第 t 小节的原始 NOTE 事件直接送入下一小节的 Note Decoder；其总结则延迟一个时间步，在生成 t+2 小节时进入 Time Decoder。主动画省略这项路由细节，以突出二维自回归。',detailBody4:'每行对应一个小节，每个横向步对应一个音乐事件，并不一定是单个音符。图中的音符仅用于示意；实际事件还可以表示节拍、乐器等结构。STO 表示小节结束。4 个小节及其事件数量只是演示，并非模型限制。',source1:'代码：隐藏状态与前缀',source2:'代码：历史更新',footer:'Decoder-in-Decoder · 可视化原理',shortcuts:'空格播放 · → 单步 · R 重新开始'
};
const tr = (en,cn)=>lang==='en'?en:cn;
const text = (x,y,s,fill=C.ink,size=18,extra='')=>`<text x="${x}" y="${y}" fill="${fill}" font-size="${size}" ${extra}>${s}</text>`;
const rect = (x,y,w,h,fill,stroke='none',rx=8,extra='')=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" ${extra}/>`;
const path = (d,color,width=1.5,extra='')=>`<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" ${extra}/>`;
const arrow=(x1,y1,x2,y2,color,active=false)=>path(`M${x1} ${y1}L${x2} ${y2}`,color,1.6,`marker-end="url(#${color===C.teal?'at':color===C.amber?'aa':'ap'})" ${active?'class="flow-active"':''}`);
const definitions = `<defs>${[['ap',C.purple],['at',C.teal],['aa',C.amber]].map(([id,c])=>`<marker id="${id}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L8 4L0 8Z" fill="${c}"/></marker>`).join('')}</defs>`;
const note=(x,y,p,color,scale=1)=>`<g transform="translate(${x} ${y-p*3}) scale(${scale})" fill="${color}" stroke="${color}" stroke-width="1.5"><ellipse rx="5.5" ry="3.8" transform="rotate(-23)"/><path d="M5 -1V-25" fill="none"/></g>`;
const sub=n=>String(n).split('').map(x=>'₀₁₂₃₄₅₆₇₈₉'[x]).join('');
function shownCount(row,s){return row<s.t||s.kind==='done'?score[row].length:row===s.t?s.n:0;}
function rowFinished(row,s){return row<s.t||s.kind==='done'||(row===s.t&&['stop','summary','feedback'].includes(s.kind));}
function summarized(row,s){return row<s.t||s.kind==='done'||(row===s.t&&['summary','feedback'].includes(s.kind));}

function renderDecoder(s){
  const done=s.kind==='done',predict=s.kind==='predict',inject=s.kind==='inject',event=s.kind==='event',feedback=s.kind==='feedback',sum=s.kind==='summary';
  const t=s.t+1, tone=event||s.kind==='stop'?C.teal:C.purple;
  let out=`<svg viewBox="0 0 900 534" role="img" aria-label="${tr('Two-dimensional decoding: horizontal musical events, vertical bars, latent guidance and summary feedback.','二维解码：横向音乐事件，纵向小节，隐藏状态指导与总结反馈。')}">${definitions}`;
  // Latent guidance is deliberately the first and most visible causal chain.
  out+=rect(20,2,860,113,'#faf9fd','#eeeaf6',10);
  out+=text(37,24,tr('HISTORY → PREDICTED LATENT GUIDANCE → EVENTS','历史 → 预测隐空间指导 → 音乐事件'),C.muted,12,'letter-spacing="1"');
  out+=rect(36,43,113,40,'#fff',C.line,7)+text(92,68,tr('History','已有历史'),C.muted,17,'text-anchor="middle"');
  out+=arrow(151,63,174,63,C.purple,predict||feedback);
  out+=rect(180,43,143,40,predict?C.purple:C.pl,predict?C.purple:'#d5c6f5',7)+text(251,68,'Time Decoder',predict?'#fff':C.purple,17,'text-anchor="middle" font-weight="600"');
  out+=text(251,101,done?tr('complete','完成'):predict?tr('predicting guidance','预测本小节指导'):tr('waiting for this bar','等待本小节完成'),predict?C.purple:C.muted,13,'text-anchor="middle"');
  out+=arrow(326,63,351,63,C.purple,predict);
  out+=rect(359,38,57,50,C.pl,C.purple,8,`stroke-width="${predict?2.5:1}"`)+text(388,72,`h${sub(t)}`,C.purple,27,'class="math" text-anchor="middle"');
  out+=text(388,101,tr('hidden state','隐藏状态'),C.purple,13,'text-anchor="middle"');
  out+=arrow(419,63,453,63,C.purple,inject)+text(436,44,'MLP',C.muted,10,'text-anchor="middle"');
  for(let q=0;q<4;q++){out+=rect(463+q*19,43,14,40,inject?C.purple:'#a78bdd','none',4);out+=path(`M${467+q*19} 54v18`,'#fff',2);}
  out+=text(498,101,`P${sub(t)} · `+tr('latent prefix','隐状态前缀'),C.purple,13,'text-anchor="middle"');
  out+=arrow(545,63,591,63,C.purple,inject)+text(568,44,tr('inject','注入'),C.purple,11,'text-anchor="middle"');
  out+=rect(599,43,146,40,event?C.teal:C.tl,event?C.teal:'#a9d7cf',7)+text(672,68,'Note Decoder',event?'#fff':C.teal,17,'text-anchor="middle" font-weight="600"');
  out+=arrow(750,63,778,63,C.teal,event)+note(807,75,2,event?C.teal:C.muted,1.1)+note(843,75,4,event?C.teal:C.muted,1.1);
  out+=text(672,101,event?tr('generating events','逐事件生成'):done?tr('complete','完成'):tr('conditioned on P','接受前缀指导')+(lang==='en'?sub(t):''),C.teal,13,'text-anchor="middle"');

  // The two axes frame the raster order, rather than showing a generic pipeline.
  out+=text(288,146,tr('NOTE AXIS','事件维度'),C.teal,13,'letter-spacing="1.5" font-weight="600"');
  out+=text(438,146,tr('Autoregression within each bar','每个小节内的自回归'),C.muted,14);
  out+=arrow(288,157,762,157,C.teal);
  out+=text(139,145,tr('Guidance','隐空间指导'),C.purple,15,'text-anchor="middle"');
  out+=text(826,145,tr('Summary','总结'),C.muted,15,'text-anchor="middle"');
  out+=arrow(47,185,47,477,C.purple);
  out+=text(23,336,tr('TIME AXIS · BAR t','时间维度 · 小节 t'),C.purple,14,'transform="rotate(-90 23 336)" text-anchor="middle" letter-spacing="1"');

  score.forEach((notes,r)=>{
    const y=201+r*82,active=r===s.t&&!done,past=r<s.t||done,count=shownCount(r,s),finished=rowFinished(r,s),summary=summarized(r,s),n=notes.length;
    out+=rect(67,y-22,798,65,active?'#f7f5fe':past?'#fcfcfe':'#fff',active?'#e0d5f6':'#eff0f5',9);
    out+=text(86,y+16,`t=${r+1}`,active?C.purple:C.muted,18,'class="math"');
    const opacity=past||active?1:.33;
    out+=`<g opacity="${opacity}">`;
    out+=rect(130,y-8,42,38,active?C.pl:'#f2eff9',active?C.purple:'#d8cceb',6)+text(151,y+18,`h${sub(r+1)}`,C.purple,22,'class="math" text-anchor="middle"');
    out+=arrow(175,y+10,189,y+10,C.purple,active&&inject);
    for(let q=0;q<4;q++)out+=rect(194+q*10,y-5,7,30,C.purple,'none',2);
    out+=arrow(239,y+10,270,y+10,C.purple,active&&(inject||event));
    out+='</g>';
    // Shared event slots visibly have different stopping positions across bars.
    for(let j=0;j<7;j++){
      const x=285+j*64,exists=j<count,isStop=j===n&&finished;
      out+=rect(x,y-11,51,42,exists?C.tl:isStop?C.al:'#fafbfd',exists?'#b5ded6':isStop?'#e4c891':'#eceef3',5);
      if(exists){out+=`<g class="${active&&event&&j===count-1?'token-new':''}">`;for(let k=0;k<5;k++)out+=path(`M${x+5} ${y+2+k*4}h41`,'#bed9d3',.7);out+=note(x+25,y+20,notes[j],C.teal,.88)+text(x+25,y+53,`x${sub(r+1)},${sub(j+1)}`,C.teal,14,'text-anchor="middle" class="math"')+'</g>';}
      else if(isStop){out+=text(x+25,y+15,'STO',C.amber,13,'text-anchor="middle" font-weight="600"');}
      else if(j===count&&active&&(event||inject)){out+=rect(x+17,y+5,17,3,'#bdb0df','none',1,'class="pulse"');}
      if(j<n&&j<count-1)out+=arrow(x+53,y+10,x+61,y+10,C.teal);
    }
    if(summary){out+=path(`M${285+n*64+53} ${y+10}H795`,C.amber,1.5,active&&sum?'class="flow-active"':'');out+=rect(803,y-8,46,38,C.al,'#e4c891',8)+text(826,y+18,`s${sub(r+1)}`,C.amber,22,'class="math" text-anchor="middle"');}
    else out+=text(826,y+16,'—','#d0d4de',18,'text-anchor="middle"');
    if(active&&feedback&&r<3){out+=path(`M826 ${y+31}V${y+65}H151V${y+72}`,C.purple,2,'marker-end="url(#ap)" class="flow-active"');out+=rect(383,y+55,239,22,'#fff','none',5)+text(502,y+71,tr('feedback · unlock next bar','总结反馈 · 允许进入下一小节'),C.purple,12,'text-anchor="middle"');}
    if(active&&(event||inject)){out+=text(758,y+53,tr('t stays fixed','t 保持不变'),C.purple,12,'text-anchor="end"');}
  });
  out+=text(87,518,done?tr('All four bars complete. Both autoregressive dimensions are visible.','四个小节生成完成。两个自回归维度共同构成乐谱。'):tr('One row = one bar. Finish the row before moving down.','每行 = 一个小节。完成当前行，才能向下推进。'),C.muted,15);
  out+='</svg>';
  $('decoder-stage').innerHTML=out;
}

function initPrinter(){
  let out=`<svg viewBox="0 0 400 528" role="img" aria-label="${tr('A printer moves horizontally to print each musical row, then advances vertically.','打印头横向打印当前行，完成后才纵向进入下一行。')}"><defs><linearGradient id="machine" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#f4f5f9"/><stop offset="1" stop-color="#e4e7ef"/></linearGradient><filter id="shadow" x="-30%" y="-30%" width="160%" height="170%"><feDropShadow dx="0" dy="6" stdDeviation="7" flood-color="#29324e" flood-opacity=".08"/></filter></defs>`;
  out+=rect(43,38,314,383,'url(#machine)','#d9dee8',17,'filter="url(#shadow)"');
  out+=rect(62,65,276,324,'#d7dce6','none',8);
  out+=rect(76,48,248,369,'#fff','#dde1ea',3,'filter="url(#shadow)"');
  out+=text(96,77,'DID',C.ink,12,'letter-spacing="3" font-weight="600"');
  out+=text(304,77,tr('AN UNFOLDING SCORE','正在生成的乐谱'),C.muted,8,'letter-spacing=".5" text-anchor="end"');
  for(let r=0;r<4;r++){
    const y=125+r*69;
    out+=`<g id="paper-row-${r}">`+rect(85,y-24,230,55,'transparent','none',4)+text(95,y+18,String(r+1),'#9ca3b4',10);
    for(let k=0;k<5;k++)out+=path(`M111 ${y+k*6}H306`,'#d5dae4',.8);
    out+=path(`M111 ${y}v24M306 ${y}v24`,'#b7becd',1);
    out+=`<g id="paper-notes-${r}"></g></g>`;
  }
  // A gantry provides a literal print-head mechanism, following the same row index.
  out+=`<g id="gantry" class="head" transform="translate(0 125)">`;
  out+=rect(48,-31,303,7,'#bac2d1','none',3)+rect(48,-28,303,2,'#e8ebf2','none',0);
  out+=`<g id="printhead" class="head" transform="translate(127 0)">`+rect(-16,-47,32,35,C.ink,'none',6)+rect(-8,-39,16,8,'#5d6378','none',2)+rect(-6,-12,12,10,C.teal,'none',2)+path('M0 -1V24',C.teal,1.5,'stroke-dasharray="3 3" opacity=".7"')+`<circle cx="0" cy="24" r="3" fill="${C.teal}"/></g></g>`;
  out+=rect(43,400,314,49,'url(#machine)','#d4dae5',11)+rect(119,411,163,7,'#b9c1d0','none',3)+rect(126,412,148,3,'#656e82','none',1);
  out+=`<circle cx="68" cy="423" r="4" fill="${C.teal}"/>`+text(80,427,'DID / 2D',C.muted,10,'letter-spacing="1"');
  out+=text(200,480,tr('Print a row. Then advance.','一行打印完，再进入下一行。'),C.ink,17,'text-anchor="middle"');
  out+=text(200,505,tr('Horizontal events · vertical bars','横向生成事件 · 纵向推进小节'),C.muted,13,'text-anchor="middle"');
  out+='</svg>';
  $('printer-stage').innerHTML=out;
}
function renderPrinter(s){
  score.forEach((notes,r)=>{
    const count=shownCount(r,s),y=125+r*69;
    $('paper-notes-'+r).innerHTML=notes.slice(0,count).map((p,n)=>note(134+n*28,y+23,p,C.teal,.85)).join('')+(rowFinished(r,s)?path(`M301 ${y}v24`,C.teal,2):'');
    $('paper-row-'+r).firstChild.setAttribute('fill',r===s.t&&s.kind!=='done'?'#f0ebff':'transparent');
  });
  const x=s.kind==='predict'||s.kind==='inject'?125:Math.min(300,134+Math.max(0,s.n-1)*28);
  $('gantry').setAttribute('transform',`translate(0 ${125+s.t*69})`);
  $('printhead').setAttribute('transform',`translate(${x} 0)`);
  $('gantry').style.opacity=s.kind==='done'?'.35':'1';
}

function narration(s){
  const t=s.t+1, h=`h${sub(t)}`, p=`P${sub(t)}`, n=s.n;
  const all={
    predict:[tr('PREDICT','预测指导'),tr(`History becomes guidance for bar ${t}.`,`根据历史，产生第 ${t} 小节的隐空间指导。`),tr(`The Time Decoder produces hidden state ${h}. A learned connector turns it into the latent prefix ${p}.`,`Time Decoder 产生隐藏状态 ${h}，再通过可学习连接器转换成隐状态前缀 ${p}。`),tr('The print head is positioned for a new row.','打印头就位，准备打印新的一行。'),tr(`<strong>${h} is a predicted latent representation</strong> for the target bar, not a musical event.`,`<strong>${h} 是面向目标小节的隐空间预测表示</strong>，不是音乐事件。`)],
    inject:[tr('INJECT','注入前缀'),tr(`Inject ${p} before the first event.`,`先注入 ${p}，再生成第一个事件。`),tr(`Four latent vectors condition the Note Decoder for bar ${t}. This guidance remains available throughout the row.`,`4 个隐状态前缀向量指导第 ${t} 小节的 Note Decoder，并在整行生成过程中保持可用。`),tr('Ready to print. The paper stays on this row.','准备开始打印，纸张停留在当前行。'),tr(`<strong>${p} is a sequence of hidden vectors.</strong> It guides all events in this bar.`,`<strong>${p} 是一段隐状态向量前缀。</strong>它指导本小节内所有事件的生成。`)],
    event:[tr('GENERATE','生成事件'),tr(`Event ${n} in bar ${t}. The Time Decoder waits.`,`第 ${t} 小节的第 ${n} 个事件：Time Decoder 等待中。`),tr(`The Note Decoder predicts x${sub(t)},${sub(n)} from the latent guidance and preceding events. The bar index t does not advance.`,`Note Decoder 根据隐状态指导及此前事件预测 x${sub(t)},${sub(n)}。小节时间步 t 不前进。`),tr('<strong>Print left to right.</strong> Do not advance to the next row.','<strong>从左向右打印。</strong>当前行完成之前不换行。'),tr(`<strong>${p} stays in context.</strong> Each new event conditions the next event.`,`<strong>${p} 持续提供指导。</strong>每个新事件又参与下一事件的预测。`)],
    stop:[tr('END BAR','小节结束'),tr(`STO closes bar ${t}. Only now is the row complete.`,`STO 结束第 ${t} 小节，当前行此时才完成。`),tr('The Note Decoder emits the stop event. No events from the following bar were generated in parallel.','Note Decoder 输出小节结束事件。下一小节的事件尚未开始生成。'),tr('<strong>The row is finished.</strong> The next row can now be prepared.','<strong>当前行打印完成。</strong>现在才可以准备下一行。'),tr('<strong>STO is the boundary</strong> between within-bar generation and the next time step.','<strong>STO 是边界：</strong>当前小节的事件生成结束，才允许准备下一步。')],
    summary:[tr('SUMMARIZE','得到总结'),tr(`Completed events become one summary s${sub(t)}.`,`已完成的事件，汇总为 s${sub(t)}。`),tr('Compress the completed bar into a summary. Its internal encoder details are omitted here.','将完整小节压缩为一个总结；这里省略总结编码器的内部细节。'),tr('The completed row is held before advancing.','保留已经打印完成的这一行，等待进入下一行。'),tr('<strong>One completed bar → one summary.</strong> Only the information flow matters here.','<strong>一个完整小节 → 一个总结。</strong>这里仅展示信息如何流动。')],
    feedback:[tr('FEEDBACK','历史反馈'),t<4?tr(`Feed back the summary. Unlock bar ${t+1}.`,`总结反馈，允许进入第 ${t+1} 小节。`):tr('The final bar joins the completed history.','最后一个小节并入已完成的历史。'),t<4?tr(`In this conceptual cycle, s${sub(t)} updates the history used to predict guidance for the next bar.`,`在概念循环中，s${sub(t)} 更新历史，用于预测下一小节的隐藏状态指导。`):tr('The same cycle can continue for more bars. This demonstration ends after four.','同一个循环可以继续生成更多小节。本演示在四个小节后结束。'),tr('<strong>Advance only after finishing the row.</strong> The head returns to the left.','<strong>当前行完成后才能换行。</strong>随后打印头回到左端。'),tr('<strong>Vertical autoregression:</strong> completed music informs the next bar’s guidance.','<strong>纵向自回归：</strong>已生成音乐参与后续小节指导信息的预测。')],
    done:[tr('COMPLETE','生成完成'),tr('Two nested autoregressive loops. One score.','两个嵌套的自回归循环，完成一张乐谱。'),tr('Across bars: predict latent guidance. Within a bar: generate events until STO. Repeat, in order.','跨小节：预测隐状态指导。小节内：逐事件生成直到 STO。按此顺序循环。'),tr('The score is complete, one row at a time.','逐行完成，最终得到完整乐谱。'),tr('<strong>Time advances down. Events unfold across.</strong> Each bar has its own latent guidance.','<strong>时间向下推进，事件横向展开。</strong>每个小节都有自己的隐状态指导。')]
  };
  return all[s.kind];
}
function render(){
  const s=phases[index],copy=narration(s);
  renderDecoder(s);renderPrinter(s);
  ['phase-pill','narration-title','narration-text'].forEach((id,i)=>$(id).textContent=copy[i]);
  $('printer-caption').innerHTML=copy[3];$('decoder-caption').innerHTML=copy[4];
  $('coordinate').textContent=s.kind==='done'?'t = 4':`t = ${s.t+1}${s.kind==='event'?`, n = ${s.n}`:''}`;
  $('bar-counter').textContent=tr('BAR ','小节 ')+String(s.t+1).padStart(2,'0')+' / 04';
  $('timeline').value=index;$('timeline').setAttribute('aria-valuetext',copy[1]);
  updateControls();
}
function updateControls(){
  document.body.classList.toggle('paused',!playing);
  $('play-icon').textContent=playing?'Ⅱ':index===phases.length-1?'↻':'▶';
  $('play-label').textContent=playing?tr('Pause','暂停'):index===phases.length-1?tr('Replay','重播'):tr('Play','播放');
  $('play').setAttribute('aria-label',$('play-label').textContent);
  $('loop').setAttribute('aria-pressed',String(looping));
}
function schedule(){clearTimeout(timer);if(playing)timer=setTimeout(advance,phases[index].duration/speed);}
function advance(){
  if(index<phases.length-1){index++;render();if(index===phases.length-1&&!looping){playing=false;updateControls();}else schedule();}
  else if(looping){index=0;render();schedule();}else{playing=false;updateControls();}
}
function togglePlay(){if(!playing&&index===phases.length-1)index=0;playing=!playing;render();schedule();}
function restart(){index=0;render();schedule();}
function step(delta=1){playing=false;clearTimeout(timer);index=Math.max(0,Math.min(phases.length-1,index+delta));render();}
function setLanguage(){
  document.documentElement.lang=lang==='en'?'en':'zh-CN';
  for(const el of document.querySelectorAll('[data-i18n]'))el.innerHTML=(lang==='en'?initialEnglish:zh)[el.dataset.i18n];
  $('language').textContent=lang==='en'?'中文':'EN';$('language').setAttribute('aria-label',tr('Switch to Chinese','切换至英文'));
  const labels={restart:tr('Restart','重新开始'),step:tr('Next step','下一步'),fullscreen:tr('Toggle fullscreen','切换全屏'),timeline:tr('Animation progress','动画进度'),speed:tr('Playback speed','播放速度')};
  Object.entries(labels).forEach(([id,label])=>{$(id).setAttribute('aria-label',label);$(id).title=label;});
  initPrinter();render();
}
$('timeline').max=phases.length-1;
$('play').addEventListener('click',togglePlay);
$('restart').addEventListener('click',restart);
$('step').addEventListener('click',()=>step());
$('timeline').addEventListener('input',e=>{playing=false;clearTimeout(timer);index=Number(e.target.value);render();});
$('speed').addEventListener('change',e=>{speed=Number(e.target.value);schedule();});
$('loop').addEventListener('click',()=>{looping=!looping;updateControls();});
$('language').addEventListener('click',()=>{lang=lang==='en'?'zh':'en';setLanguage();});
$('fullscreen').addEventListener('click',async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch{ $('fullscreen').title=tr('Fullscreen is unavailable in this viewer.','当前预览器不支持全屏。');}});
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','BUTTON','A','SUMMARY'].includes(document.activeElement.tagName)||e.ctrlKey||e.metaKey||e.altKey)return;if(e.code==='Space'){e.preventDefault();togglePlay();}else if(e.key==='ArrowRight'){e.preventDefault();step();}else if(e.key==='ArrowLeft'){e.preventDefault();step(-1);}else if(e.key.toLowerCase()==='r')restart();});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&playing){playing=false;clearTimeout(timer);updateControls();}});
setLanguage();
if(!window.matchMedia('(prefers-reduced-motion: reduce)').matches){playing=true;updateControls();schedule();}
