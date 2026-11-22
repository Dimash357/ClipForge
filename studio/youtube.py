import re,urllib.parse,requests
from django.core.exceptions import ValidationError
HOSTS={'youtube.com','www.youtube.com','m.youtube.com','youtu.be','www.youtu.be','youtube-nocookie.com','www.youtube-nocookie.com'}
def parse_youtube_url(url):
 try:
  parsed=urllib.parse.urlsplit(url.strip())
  if parsed.scheme!='https' or parsed.hostname not in HOSTS or parsed.username or parsed.password or parsed.port not in (None,443):raise ValueError()
  if parsed.hostname in ('youtu.be','www.youtu.be'): video_id=parsed.path.strip('/').split('/')[0]
  elif parsed.path=='/watch':video_id=urllib.parse.parse_qs(parsed.query).get('v',[''])[0]
  elif parsed.path.startswith(('/shorts/','/embed/','/live/')):video_id=parsed.path.split('/')[2]
  else:raise ValueError()
  if not re.fullmatch(r'[A-Za-z0-9_-]{11}',video_id):raise ValueError()
  return video_id
 except (ValueError,TypeError):raise ValidationError('Введите корректную HTTPS-ссылку YouTube.')
def get_metadata(url):
 video_id=parse_youtube_url(url)
 # YouTube oEmbed is metadata-only. No arbitrary network URL is fetched.
 response=requests.get('https://www.youtube.com/oembed',params={'url':'https://www.youtube.com/watch?v='+video_id,'format':'json'},timeout=12)
 response.raise_for_status()
 data=response.json()
 return {'id':video_id,'title':data.get('title',''),'author':data.get('author_name',''),'thumbnail':f'https://i.ytimg.com/vi/{video_id}/hqdefault.jpg'}
