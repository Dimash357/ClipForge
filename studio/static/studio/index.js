function cookie(n){return document.cookie.split('; ').find(x=>x.startsWith(n+'='))?.split('=')[1]||''}
const csrf=()=>decodeURIComponent(cookie('csrftoken'));
async function post(url,body){return fetch(url,{method:'POST',headers:{'Content-Type':'application/json','X-CSRFToken':csrf()},body:JSON.stringify(body)})}
async function lookup(){
  const el=document.getElementById('meta'),st=document.getElementById('source_status'),url=document.getElementById('youtube_url').value;
  el.textContent='Проверяем ссылку...';
  try{
    const r=await post('/api/youtube/metadata/',{url});const j=await r.json();if(!r.ok)throw Error(j.error);
    el.textContent='';const img=document.createElement('img');img.src=j.thumbnail;img.alt='YouTube thumbnail';
    const txt=document.createElement('div');txt.textContent=j.title+' · '+j.author;el.append(img,txt);
    const sr=await post('/api/youtube/source-check/',{url});const s=await sr.json();
    st.textContent=s.available?'✓ Локальный исходник найден: '+s.filename+' — загружать файл не нужно.':'Локального исходника нет. Программа попробует получить видео по ссылке (если включена опция).';
  }catch(e){el.textContent='Не удалось получить метаданные: '+e.message}
}
document.getElementById('lookupbtn').addEventListener('click',lookup);
