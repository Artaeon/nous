package server

// webUI is the full-featured web interface with login, tabbed navigation,
// chat history, dashboard, memory viewer, tools catalog, and settings.
const webUI = `<!DOCTYPE html>
<html lang="en">
<head>
<title>Nous</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--bg4:#30363d;--fg:#e6edf3;--fg2:#8b949e;--fg3:#484f58;--accent:#58a6ff;--accent2:#1f6feb;--green:#3fb950;--yellow:#d29922;--red:#f85149;--cyan:#79c0ff;--magenta:#d2a8ff;--border:#30363d;--font:-apple-system,BlinkMacSystemFont,'SF Pro Text','Segoe UI',system-ui,sans-serif;--mono:'SF Mono','Fira Code','JetBrains Mono',Consolas,monospace;--t:.15s ease}
*{margin:0;padding:0;box-sizing:border-box}html,body{height:100%;overflow:hidden}
body{font-family:var(--mono);background:var(--bg);color:var(--fg);font-size:13px;-webkit-font-smoothing:antialiased}

/* Login */
.login{position:fixed;inset:0;background:var(--bg);display:flex;align-items:center;justify-content:center;z-index:100}.login.hidden{display:none}
.login-box{width:360px;text-align:center}
.login-box h1{font-size:28px;font-weight:300;letter-spacing:8px;color:var(--fg);margin-bottom:4px}
.login-box p{color:var(--fg2);font-size:12px;margin-bottom:32px;font-family:var(--font)}
.login-box input{width:100%;padding:12px 16px;background:var(--bg2);border:1px solid var(--border);color:var(--fg);border-radius:8px;font-family:var(--mono);font-size:14px;text-align:center;outline:none;transition:border var(--t)}
.login-box input:focus{border-color:var(--accent)}.login-box input::placeholder{color:var(--fg3)}
.login-box button{width:100%;margin-top:12px;padding:10px;background:var(--accent2);color:#fff;border:none;border-radius:8px;font-family:var(--font);font-size:14px;font-weight:600;cursor:pointer;transition:background var(--t)}
.login-box button:hover{background:var(--accent)}
.login-box .skip{margin-top:12px;font-size:11px;color:var(--fg3);cursor:pointer;font-family:var(--font)}.login-box .skip:hover{color:var(--fg2)}
.login-err{color:var(--red);font-size:12px;margin-top:8px;min-height:18px;font-family:var(--font)}

.app{display:flex;height:100vh;flex-direction:column}.app.hidden{display:none}

/* Topbar */
.topbar{display:flex;align-items:center;padding:0 16px;height:40px;background:var(--bg2);border-bottom:1px solid var(--border);flex-shrink:0;gap:12px}
.topbar .logo{color:var(--green);font-weight:700;font-size:13px;letter-spacing:2px}
.topbar .sep{color:var(--fg3)}
.topbar .model-info{color:var(--fg2);font-size:12px}
.topbar .tabs{display:flex;gap:2px;margin-left:24px}
.topbar .tab{padding:6px 14px;border-radius:6px;cursor:pointer;transition:all var(--t);color:var(--fg2);font-size:12px;font-family:var(--font);font-weight:500}
.topbar .tab:hover{background:var(--bg3);color:var(--fg)}
.topbar .tab.active{background:var(--bg3);color:var(--fg)}
.topbar .spacer{flex:1}
.topbar .dot{width:6px;height:6px;border-radius:50%;background:var(--green);display:inline-block;margin-right:4px;animation:pulse 2s ease infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.topbar .status-text{font-size:11px;color:var(--fg3)}

/* Main */
.main{flex:1;display:flex;min-height:0;overflow:hidden}
.panel{flex:1;display:none;flex-direction:column;overflow:hidden}.panel.active{display:flex}

/* Chat panel */
.chat-layout{display:flex;flex:1;min-height:0}
.chat-side{width:240px;border-right:1px solid var(--border);background:var(--bg2);display:flex;flex-direction:column;overflow:hidden;flex-shrink:0}
.chat-side h3{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--fg3);padding:12px 12px 8px;font-weight:600}
.chat-side-scroll{flex:1;overflow-y:auto;padding:0 8px 8px}
.chat-side-scroll::-webkit-scrollbar{width:4px}.chat-side-scroll::-webkit-scrollbar-thumb{background:var(--bg4);border-radius:2px}
.task-item{display:flex;align-items:flex-start;gap:6px;padding:5px 8px;border-radius:6px;margin-bottom:2px;font-size:11px;cursor:pointer;transition:background var(--t)}
.task-item:hover{background:var(--bg3)}
.task-item .check{width:12px;height:12px;border:1.5px solid var(--fg3);border-radius:50%;flex-shrink:0;margin-top:2px;cursor:pointer;transition:all var(--t)}
.task-item .check:hover{border-color:var(--green);background:rgba(63,185,80,.15)}
.task-item .text{flex:1;line-height:1.4}
.task-item .due{font-size:9px;color:var(--fg3)}.task-item .due.overdue{color:var(--red)}
.job-card{padding:6px 8px;border:1px solid var(--border);border-radius:6px;margin-bottom:4px;background:var(--bg);font-size:11px}
.job-card .jmeta{display:flex;justify-content:space-between;margin-bottom:2px}
.pill{font-size:9px;font-weight:700;text-transform:uppercase;padding:1px 6px;border-radius:8px}
.pill.queued{background:#3d2e00;color:var(--yellow)}.pill.running{background:#0c2d6b;color:var(--cyan)}.pill.completed{background:#0d3117;color:var(--green)}.pill.failed,.pill.canceled{background:#3c1111;color:var(--red)}
.side-empty{color:var(--fg3);font-size:11px;text-align:center;padding:12px 0}

.chat-main{flex:1;display:flex;flex-direction:column;min-width:0}
.output{flex:1;overflow-y:auto;padding:16px 20px}.output::-webkit-scrollbar{width:6px}.output::-webkit-scrollbar-thumb{background:var(--bg4);border-radius:3px}
.line{margin-bottom:10px;animation:fadeUp .2s ease;max-width:900px}
@keyframes fadeUp{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
.line .prompt{color:var(--green);font-weight:600}.line .cmd{color:var(--fg)}.line .out{color:var(--fg2);white-space:pre-wrap;word-break:break-word;margin-top:4px;padding:4px 0 4px 12px;border-left:2px solid var(--bg4)}
.line .out.err{color:var(--red);border-left-color:var(--red)}.line .time{color:var(--fg3);font-size:11px;margin-top:2px}.line .sys{color:var(--cyan);font-style:italic}
.thinking-dots{display:inline-flex;gap:3px;margin-left:4px}.thinking-dots span{width:5px;height:5px;border-radius:50%;background:var(--fg3);animation:bounce 1.4s ease infinite}
.thinking-dots span:nth-child(2){animation-delay:.15s}.thinking-dots span:nth-child(3){animation-delay:.3s}
@keyframes bounce{0%,80%,100%{transform:scale(.5);opacity:.3}40%{transform:scale(1);opacity:1}}
.welcome{text-align:center;padding:60px 20px;color:var(--fg3)}.welcome h2{font-size:18px;font-weight:300;letter-spacing:4px;color:var(--fg2);margin-bottom:8px;font-family:var(--font)}
.welcome p{font-size:12px;line-height:1.8;max-width:400px;margin:0 auto;font-family:var(--font)}
.welcome .cmds{margin-top:24px;display:flex;flex-wrap:wrap;gap:6px;justify-content:center}
.welcome .cmds span{padding:4px 10px;border:1px solid var(--border);border-radius:4px;font-size:11px;color:var(--fg2);cursor:pointer;transition:all var(--t)}
.welcome .cmds span:hover{border-color:var(--accent);color:var(--accent);background:rgba(88,166,255,.08)}
.input-bar{padding:8px 16px 12px;background:var(--bg2);border-top:1px solid var(--border);display:flex;gap:8px;flex-shrink:0}
.input-bar input{flex:1;padding:8px 12px;background:var(--bg);border:1px solid var(--border);color:var(--fg);border-radius:6px;font-family:var(--mono);font-size:13px;outline:none;transition:border var(--t)}
.input-bar input:focus{border-color:var(--accent)}.input-bar input::placeholder{color:var(--fg3)}
.input-bar button{padding:6px 14px;border:none;border-radius:6px;font-family:var(--mono);font-size:12px;font-weight:600;cursor:pointer;transition:all var(--t)}
.input-bar .btn-go{background:var(--accent2);color:#fff}.input-bar .btn-go:hover{background:var(--accent)}
.input-bar .btn-q{background:var(--bg3);color:var(--fg2)}.input-bar .btn-q:hover{background:var(--bg4);color:var(--fg)}
.input-bar button:disabled{opacity:.3;cursor:wait}

/* Content panels (dashboard, memory, tools, settings) */
.panel-scroll{flex:1;overflow-y:auto;padding:24px 32px}.panel-scroll::-webkit-scrollbar{width:6px}.panel-scroll::-webkit-scrollbar-thumb{background:var(--bg4);border-radius:3px}
.panel-title{font-size:18px;font-weight:600;margin-bottom:20px;font-family:var(--font);color:var(--fg)}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px;margin-bottom:24px}
.card{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:16px;transition:border var(--t)}
.card:hover{border-color:var(--bg4)}
.card h4{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--fg3);margin-bottom:10px;font-weight:600;font-family:var(--font)}
.card .val{font-size:28px;font-weight:700;color:var(--fg);font-family:var(--font)}
.card .sub{font-size:12px;color:var(--fg2);margin-top:4px;font-family:var(--font)}
.card-list{list-style:none}.card-list li{padding:6px 0;border-bottom:1px solid var(--border);font-size:12px;display:flex;justify-content:space-between}
.card-list li:last-child{border:none}
.card-list .label{color:var(--fg2)}.card-list .value{color:var(--fg);font-weight:500}
.section-title{font-size:13px;font-weight:600;margin:24px 0 12px;color:var(--fg2);font-family:var(--font);text-transform:uppercase;letter-spacing:.05em}
.data-table{width:100%;border-collapse:collapse;font-size:12px;margin-bottom:24px}
.data-table th{text-align:left;padding:8px 12px;border-bottom:2px solid var(--border);color:var(--fg3);font-weight:600;text-transform:uppercase;font-size:10px;letter-spacing:.06em}
.data-table td{padding:8px 12px;border-bottom:1px solid var(--border);color:var(--fg2);font-family:var(--mono)}
.data-table tr:hover td{background:var(--bg2)}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600;background:var(--bg3);color:var(--fg2)}
.badge.green{background:#0d3117;color:var(--green)}.badge.blue{background:#0c2d6b;color:var(--cyan)}.badge.yellow{background:#3d2e00;color:var(--yellow)}
.pref-row{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;border-bottom:1px solid var(--border)}
.pref-row .k{color:var(--fg2);font-size:12px}.pref-row .v{color:var(--fg);font-size:12px}
.empty-msg{color:var(--fg3);text-align:center;padding:32px;font-family:var(--font)}

/* Bottom bar */
.bottombar{height:24px;background:var(--accent2);display:flex;align-items:center;padding:0 12px;font-size:11px;color:rgba(255,255,255,.85);gap:16px;flex-shrink:0;font-family:var(--font)}
.bottombar .spacer{flex:1}

@media(max-width:768px){.chat-side{display:none}.topbar .tabs{display:none}.cards{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="login" id="login"><div class="login-box">
  <h1>NOUS</h1><p>Your personal AI assistant</p>
  <input type="password" id="keyInput" placeholder="API Key" autofocus>
  <button onclick="doLogin()">Connect</button>
  <div class="skip" onclick="skipLogin()">Connect without key (local mode)</div>
  <div class="login-err" id="loginErr"></div>
</div></div>

<div class="app hidden" id="app">
<div class="topbar">
  <span class="logo">NOUS</span><span class="sep">|</span><span class="model-info" id="modelInfo">-</span>
  <div class="tabs">
    <div class="tab active" onclick="switchTab('chat')">Chat</div>
    <div class="tab" onclick="switchTab('dashboard')">Dashboard</div>
    <div class="tab" onclick="switchTab('memory')">Memory</div>
    <div class="tab" onclick="switchTab('tools')">Tools</div>
    <div class="tab" onclick="switchTab('settings')">Settings</div>
  </div>
  <div class="spacer"></div>
  <span class="status-text"><span class="dot"></span><span id="uptimeText">-</span></span>
</div>
<div class="main">
  <!-- CHAT TAB -->
  <div class="panel active" id="tab-chat">
    <div class="chat-layout">
      <div class="chat-side">
        <h3>Tasks</h3>
        <div class="chat-side-scroll" id="taskList"><div class="side-empty">No tasks</div></div>
        <h3>Background Jobs</h3>
        <div class="chat-side-scroll" id="jobList"><div class="side-empty">No jobs</div></div>
      </div>
      <div class="chat-main">
        <div class="output" id="output">
          <div class="welcome"><h2>Welcome</h2><p>Fully local AI. Everything private.<br>Type a message or try a command.</p>
            <div class="cmds" id="quickCmds">
              <span onclick="runCmd('/briefing')">/briefing</span><span onclick="runCmd('/today')">/today</span>
              <span onclick="runCmd('/now')">/now</span><span onclick="runCmd('/tasks')">/tasks</span>
              <span onclick="runCmd('/compass')">/compass</span><span onclick="runCmd('/dashboard')">/dashboard</span>
              <span onclick="runCmd('/help')">/help</span>
            </div>
          </div>
        </div>
        <div class="input-bar">
          <input type="text" id="chatInput" placeholder="nous >" disabled>
          <button class="btn-go" id="btnGo" onclick="handleInput()" disabled>Run</button>
          <button class="btn-q" id="btnQ" onclick="handleQueue()" disabled>Queue</button>
        </div>
      </div>
    </div>
  </div>

  <!-- DASHBOARD TAB -->
  <div class="panel" id="tab-dashboard">
    <div class="panel-scroll">
      <div class="panel-title">Dashboard</div>
      <div class="cards" id="dashCards"></div>
      <div class="section-title">Training Quality</div>
      <div id="dashTraining" class="card" style="max-width:500px"></div>
    </div>
  </div>

  <!-- MEMORY TAB -->
  <div class="panel" id="tab-memory">
    <div class="panel-scroll">
      <div class="panel-title">Memory</div>
      <div class="section-title">Long-Term Memory (Persistent Facts)</div>
      <table class="data-table" id="ltmTable"><thead><tr><th>Key</th><th>Value</th><th>Category</th><th>Accessed</th></tr></thead><tbody></tbody></table>
      <div class="section-title">Working Memory (Active Context)</div>
      <table class="data-table" id="wmTable"><thead><tr><th>Key</th><th>Value</th><th>Relevance</th></tr></thead><tbody></tbody></table>
      <div class="section-title">Episodic Memory (Recent Interactions)</div>
      <table class="data-table" id="epTable"><thead><tr><th>Time</th><th>Input</th><th>Output</th><th>Duration</th></tr></thead><tbody></tbody></table>
    </div>
  </div>

  <!-- TOOLS TAB -->
  <div class="panel" id="tab-tools">
    <div class="panel-scroll">
      <div class="panel-title">Tools &amp; Commands</div>
      <div class="section-title">Built-in Tools</div>
      <table class="data-table" id="toolsTable"><thead><tr><th>Name</th><th>Description</th></tr></thead><tbody></tbody></table>
      <div class="section-title">Slash Commands</div>
      <div class="cards" id="cmdCards"></div>
    </div>
  </div>

  <!-- SETTINGS TAB -->
  <div class="panel" id="tab-settings">
    <div class="panel-scroll">
      <div class="panel-title">Settings</div>
      <div class="section-title">Preferences</div>
      <div class="card" id="prefsCard"></div>
      <div class="section-title">Sessions</div>
      <table class="data-table" id="sessTable"><thead><tr><th>Name</th><th>Messages</th><th>Last Updated</th><th>ID</th></tr></thead><tbody></tbody></table>
      <div class="section-title">Conversation History</div>
      <table class="data-table" id="convTable"><thead><tr><th>Role</th><th>Content</th></tr></thead><tbody></tbody></table>
      <div class="section-title">Connection</div>
      <div class="card"><ul class="card-list">
        <li><span class="label">API Key</span><span class="value" id="settKeyStatus">-</span></li>
        <li><span class="label">Action</span><span class="value"><span style="color:var(--red);cursor:pointer" onclick="logout()">Disconnect</span></span></li>
      </ul></div>
    </div>
  </div>
</div>
<div class="bottombar">
  <span id="bbVersion">-</span><span id="bbModel">-</span><span id="bbTools">-</span>
  <span class="spacer"></span><span id="bbTasks">-</span><span id="bbJobs">-</span><span id="bbUptime">-</span>
</div>
</div>

<script>
const $=s=>document.getElementById(s);let apiKey=localStorage.getItem('nous_api_key')||'';let started=false;
function hdr(x){const h=Object.assign({'Content-Type':'application/json'},x||{});if(apiKey)h['Authorization']='Bearer '+apiKey;return h}
async function af(u,o){o=o||{};o.headers=hdr(o.headers);return fetch(u,o)}

// Login
$('keyInput').addEventListener('keydown',e=>{if(e.key==='Enter')doLogin()});
async function doLogin(){apiKey=$('keyInput').value.trim();if(!apiKey){$('loginErr').textContent='Enter API key';return}
  try{const r=await af('/api/status');if(r.status===401){$('loginErr').textContent='Invalid key';return}localStorage.setItem('nous_api_key',apiKey);enterApp()}catch(e){$('loginErr').textContent='Cannot connect'}}
function skipLogin(){apiKey='';localStorage.removeItem('nous_api_key');enterApp()}
function logout(){localStorage.removeItem('nous_api_key');location.reload()}
(async()=>{try{const r=await af('/api/health');if(r.ok){const s=await af('/api/status');if(s.ok){enterApp();return}}}catch(e){}$('login').classList.remove('hidden')})();

function enterApp(){$('login').classList.add('hidden');$('app').classList.remove('hidden');
  $('chatInput').disabled=false;$('btnGo').disabled=false;$('btnQ').disabled=false;$('chatInput').focus();
  $('settKeyStatus').textContent=apiKey?'Connected (key set)':'Local mode (no key)';
  loadAll();setInterval(loadJobs,5000);setInterval(loadTasks,10000);setInterval(loadStatus,15000)}

// Tabs
function switchTab(name){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  $('tab-'+name).classList.add('active');
  document.querySelector('.tab[onclick*="'+name+'"]').classList.add('active');
  if(name==='dashboard')loadDashboard();if(name==='memory')loadMemory();if(name==='tools')loadTools();if(name==='settings')loadSettings();
  if(name==='chat')$('chatInput').focus();
}

function loadAll(){loadStatus();loadTasks();loadJobs()}

// Status
async function loadStatus(){try{const r=await af('/api/status');const s=await r.json();
  $('modelInfo').textContent=s.model+' | '+s.tool_count+' tools';$('uptimeText').textContent=s.uptime;
  $('bbVersion').textContent='v'+s.version;$('bbModel').textContent=s.model;$('bbTools').textContent=s.tool_count+' tools';$('bbUptime').textContent=s.uptime;
  $('bbJobs').textContent=(s.running_jobs+s.queued_jobs)+' jobs'}catch(e){$('modelInfo').textContent='offline'}}

// Tasks
async function loadTasks(){try{const r=await af('/api/assistant/tasks');const d=await r.json();const tl=$('taskList');tl.innerHTML='';
  if(!d.tasks||!d.tasks.length){tl.innerHTML='<div class="side-empty">No tasks</div>';$('bbTasks').textContent='0 tasks';return}
  $('bbTasks').textContent=d.tasks.length+' tasks';
  d.tasks.forEach(t=>{const el=document.createElement('div');el.className='task-item';const due=t.due_at?new Date(t.due_at):null;const ov=due&&due<new Date();
    el.innerHTML='<div class="check" onclick="event.stopPropagation();doneTask(\''+esc(t.id)+'\')"></div><span class="text">'+esc(t.title)+'</span>'+(due?'<span class="due'+(ov?' overdue':'')+'">'+fmtDate(due)+'</span>':'');
    tl.appendChild(el)})}catch(e){}}
async function doneTask(id){await af('/api/assistant/tasks/'+id+'/done',{method:'POST'});loadTasks();addSys('Task completed.')}

// Jobs
async function loadJobs(){try{const r=await af('/api/jobs');const d=await r.json();const jl=$('jobList');jl.innerHTML='';
  if(!d.jobs||!d.jobs.length){jl.innerHTML='<div class="side-empty">No jobs</div>';return}
  d.jobs.forEach(j=>{const c=document.createElement('div');c.className='job-card';
    c.innerHTML='<div class="jmeta"><span style="color:var(--fg3)">'+esc(j.id).slice(0,8)+'</span><span class="pill '+esc(j.status)+'">'+esc(j.status)+'</span></div><div class="jmsg">'+esc(j.message)+'</div>';
    jl.appendChild(c)})}catch(e){}}

// Dashboard
async function loadDashboard(){try{const r=await af('/api/dashboard');const d=await r.json();
  $('dashCards').innerHTML=[
    card('Model',d.model||'-','Active LLM'),card('Uptime',d.uptime||'-','Since last restart'),
    card('Working Memory',d.working_memory_size||0,'Active slots'),card('Long-Term Memory',d.longterm_memory_size||0,'Persistent facts'),
    card('Episodes',d.episodes_total||0,'Success rate: '+(d.success_rate*100||0).toFixed(0)+'%'),
    card('Training Pairs',d.training_pairs||0,'For fine-tuning'),
    card('Conversation',d.conversation_messages||0,'Messages this session'),card('Tasks',d.pending_tasks||0,(d.unread_notifications||0)+' unread notifications')
  ].join('');
  const qd=d.quality_distribution||{};
  $('dashTraining').innerHTML='<h4>Quality Distribution</h4><ul class="card-list">'+Object.entries(qd).map(([k,v])=>'<li><span class="label">'+esc(k)+'</span><span class="value">'+v+'</span></li>').join('')+'</ul>';
}catch(e){$('dashCards').innerHTML='<div class="empty-msg">Unable to load dashboard</div>'}}
function card(t,v,s){return'<div class="card"><h4>'+esc(t)+'</h4><div class="val">'+esc(''+v)+'</div><div class="sub">'+esc(s)+'</div></div>'}

// Memory
async function loadMemory(){
  try{const r=await af('/api/longterm');const d=await r.json();const tb=$('ltmTable').querySelector('tbody');tb.innerHTML='';
    if(d.entries)d.entries.forEach(e=>{tb.innerHTML+='<tr><td style="color:var(--cyan)">'+esc(e.key)+'</td><td>'+esc(e.value)+'</td><td><span class="badge">'+esc(e.category)+'</span></td><td>'+e.access_count+'</td></tr>'})}catch(e){}
  try{const r=await af('/api/memory');const d=await r.json();const tb=$('wmTable').querySelector('tbody');tb.innerHTML='';
    if(d.items)d.items.forEach(e=>{tb.innerHTML+='<tr><td style="color:var(--magenta)">'+esc(e.key)+'</td><td>'+esc(''+e.value)+'</td><td>'+(e.relevance||0).toFixed(2)+'</td></tr>'})}catch(e){}
  try{const r=await af('/api/episodes');const d=await r.json();const tb=$('epTable').querySelector('tbody');tb.innerHTML='';
    if(d.episodes)d.episodes.slice(0,15).forEach(e=>{const t=new Date(e.timestamp);
      tb.innerHTML+='<tr><td style="color:var(--fg3)">'+t.toLocaleTimeString()+'</td><td>'+esc((e.input||'').slice(0,60))+'</td><td style="color:var(--fg2)">'+esc((e.output||'').slice(0,80))+'</td><td>'+e.duration_ms+'ms</td></tr>'})}catch(e){}}

// Tools
async function loadTools(){
  try{const r=await af('/api/tools');const d=await r.json();const tb=$('toolsTable').querySelector('tbody');tb.innerHTML='';
    if(d.tools)d.tools.forEach(t=>{tb.innerHTML+='<tr><td style="color:var(--green);font-weight:600">'+esc(t.name)+'</td><td style="color:var(--fg2)">'+esc(t.description)+'</td></tr>'})}catch(e){}
  const cmds=[{n:'/briefing',d:'Morning briefing with overdue and schedule'},{n:'/today',d:'Notifications and upcoming tasks'},
    {n:'/now',d:'What to do next'},{n:'/compass',d:'Triage: do now, focus, risks'},{n:'/tasks',d:'List pending tasks'},
    {n:'/remind',d:'Create a reminder'},{n:'/done',d:'Mark task completed'},{n:'/routines',d:'List routines'},
    {n:'/prefs',d:'View preferences'},{n:'/pref',d:'Set a preference'},{n:'/memory',d:'Working memory items'},
    {n:'/longterm',d:'Long-term memory'},{n:'/episodes',d:'Episodic memory'},{n:'/search',d:'Search episodic memory'},
    {n:'/remember',d:'Store project fact'},{n:'/recall',d:'Retrieve project fact'},{n:'/knowledge',d:'Knowledge base stats'},
    {n:'/training',d:'Training data stats'},{n:'/sessions',d:'List saved sessions'},{n:'/save',d:'Save current session'},
    {n:'/model',d:'Show active model'},{n:'/tools',d:'Tool catalog'},{n:'/dashboard',d:'System overview'},
    {n:'/help',d:'Command reference'},{n:'/status',d:'Runtime status'},{n:'/plan',d:'Delegate a task'}];
  $('cmdCards').innerHTML=cmds.map(c=>'<div class="card" style="cursor:pointer;padding:12px" onclick="switchTab(\'chat\');runCmd(\''+c.n+'\')"><h4 style="color:var(--accent);margin-bottom:4px;font-size:13px;font-family:var(--mono)">'+c.n+'</h4><div class="sub" style="font-size:11px">'+esc(c.d)+'</div></div>').join('')}

// Settings
async function loadSettings(){
  try{const r=await af('/api/assistant/preferences');const d=await r.json();const pc=$('prefsCard');
    if(d.preferences&&d.preferences.length){pc.innerHTML=d.preferences.map(p=>'<div class="pref-row"><span class="k">'+esc(p.key)+'</span><span class="v">'+esc(p.value)+'</span></div>').join('')}
    else{pc.innerHTML='<div class="empty-msg">No preferences set. Use /pref key value</div>'}}catch(e){}
  try{const r=await af('/api/sessions');const d=await r.json();const tb=$('sessTable').querySelector('tbody');tb.innerHTML='';
    if(d.sessions)d.sessions.forEach(s=>{tb.innerHTML+='<tr><td style="color:var(--fg)">'+esc(s.name)+'</td><td>'+s.message_count+'</td><td style="color:var(--fg3)">'+new Date(s.updated_at).toLocaleString()+'</td><td style="color:var(--fg3);font-size:10px">'+esc(s.id).slice(0,12)+'</td></tr>'})}catch(e){}
  try{const r=await af('/api/conversation');const d=await r.json();const tb=$('convTable').querySelector('tbody');tb.innerHTML='';
    if(d.messages)d.messages.slice(-20).forEach(m=>{if(m.role==='system')return;
      tb.innerHTML+='<tr><td><span class="badge '+(m.role==='user'?'blue':'green')+'">'+esc(m.role)+'</span></td><td style="color:var(--fg2);max-width:700px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'+esc((m.content||'').slice(0,200))+'</td></tr>'})}catch(e){}}

// Chat input
$('chatInput').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey)handleInput()});
function runCmd(cmd){$('chatInput').value=cmd;handleInput()}

async function handleInput(){const msg=$('chatInput').value.trim();if(!msg)return;$('chatInput').value='';clearWelcome();
  if(msg.startsWith('/')){addPrompt(msg);if(await handleSlash(msg))return}
  addPrompt(msg);disable(true);const el=addThinking();
  try{const r=await af('/api/chat',{method:'POST',body:JSON.stringify({message:msg})});const d=await r.json();
    el.querySelector('.out').textContent=d.answer||'(no response)';if(d.duration_ms)el.querySelector('.time').textContent=d.duration_ms+'ms'}
  catch(e){el.querySelector('.out').className='out err';el.querySelector('.out').textContent='Error: '+e.message}
  disable(false)}

async function handleQueue(){const msg=$('chatInput').value.trim();if(!msg)return;$('chatInput').value='';clearWelcome();addPrompt(msg);
  try{await af('/api/jobs',{method:'POST',body:JSON.stringify({message:msg})});addSys('Queued.');loadJobs()}catch(e){addOut('Error: '+e.message,true)}$('chatInput').focus()}

async function handleSlash(cmd){const p=cmd.split(/\s+/);const c=p[0].toLowerCase();
  try{
    if(c==='/tasks'){const r=await af('/api/assistant/tasks');const d=await r.json();
      if(!d.tasks||!d.tasks.length){addOut('No pending tasks.');return true}
      addOut(d.tasks.map(t=>(t.status==='done'?'[x]':'[ ]')+' '+t.title+(t.due_at?' ('+fmtDate(new Date(t.due_at))+')':'')+' #'+t.id).join('\n'));return true}
    if(c==='/today'){const r=await af('/api/assistant/today');const d=await r.json();let o='';
      if(d.notifications&&d.notifications.length)o+='Notifications:\n'+d.notifications.map(n=>'  '+n.message).join('\n')+'\n\n';
      if(d.today&&d.today.length)o+='Today:\n'+d.today.map(t=>'  '+t.title).join('\n')+'\n\n';
      if(d.upcoming&&d.upcoming.length)o+='Upcoming:\n'+d.upcoming.map(t=>'  '+t.title+(t.due_at?' - '+fmtDate(new Date(t.due_at)):'')).join('\n');
      addOut(o||'All clear.');return true}
    if(c==='/status'){const r=await af('/api/status');const d=await r.json();
      addOut('Version:  '+d.version+'\nModel:    '+d.model+'\nTools:    '+d.tool_count+'\nUptime:   '+d.uptime+'\nPercepts: '+d.percepts+'\nGoals:    '+d.goals+'\nJobs:     '+d.running_jobs+' running, '+d.queued_jobs+' queued');return true}
    if(c==='/prefs'){const r=await af('/api/assistant/preferences');const d=await r.json();
      if(!d.preferences||!d.preferences.length){addOut('No preferences set.');return true}
      addOut(d.preferences.map(p=>p.key+': '+p.value).join('\n'));return true}
    if(c==='/pref'&&p.length>=3){await af('/api/assistant/preferences',{method:'POST',body:JSON.stringify({key:p[1],value:p.slice(2).join(' ')})});addSys('Saved: '+p[1]);return true}
    if(c==='/routines'){const r=await af('/api/assistant/routines');const d=await r.json();
      if(!d.routines||!d.routines.length){addOut('No routines.');return true}
      addOut(d.routines.map(r=>(r.enabled?'ON ':'OFF ')+r.title+' ('+r.schedule+' @ '+r.time_of_day+')').join('\n'));return true}
    if(c==='/remind'&&p.length>=2){await af('/api/assistant/tasks',{method:'POST',body:JSON.stringify({title:p.slice(1).join(' ')})});addSys('Reminder created.');loadTasks();return true}
    if(c==='/done'&&p[1]){await af('/api/assistant/tasks/'+p[1]+'/done',{method:'POST'});addSys('Done.');loadTasks();return true}
  }catch(e){addOut('Error: '+e.message,true);return true}
  return false}

function clearWelcome(){const w=document.querySelector('.welcome');if(w)w.remove()}
function addPrompt(t){const d=document.createElement('div');d.className='line';d.innerHTML='<span class="prompt">nous &gt;</span> <span class="cmd">'+esc(t)+'</span>';$('output').appendChild(d);$('output').scrollTop=$('output').scrollHeight}
function addThinking(){const d=document.createElement('div');d.className='line';d.innerHTML='<div class="out"><span class="thinking-dots"><span></span><span></span><span></span></span></div><div class="time"></div>';$('output').appendChild(d);$('output').scrollTop=$('output').scrollHeight;return d}
function addOut(t,e){const d=document.createElement('div');d.className='line';d.innerHTML='<div class="out'+(e?' err':'')+'">'+esc(t)+'</div>';$('output').appendChild(d);$('output').scrollTop=$('output').scrollHeight}
function addSys(t){const d=document.createElement('div');d.className='line';d.innerHTML='<div class="sys">'+esc(t)+'</div>';$('output').appendChild(d);$('output').scrollTop=$('output').scrollHeight}
function disable(v){$('btnGo').disabled=v;$('btnQ').disabled=v;$('chatInput').disabled=v;if(!v)$('chatInput').focus()}
function esc(t){return(t||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function fmtDate(d){const now=new Date();const diff=d-now;const days=Math.ceil(diff/864e5);if(days===0)return'today';if(days===1)return'tomorrow';if(days===-1)return'yesterday';if(days<-1)return Math.abs(days)+'d ago';if(days<7)return'in '+days+'d';return d.toLocaleDateString([],{month:'short',day:'numeric'})}
</script>
</body>
</html>`
