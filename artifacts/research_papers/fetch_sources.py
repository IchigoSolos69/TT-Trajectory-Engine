from pathlib import Path
import concurrent.futures, urllib.request, urllib.parse, json, re
root = Path(__file__).resolve().parent
targets = {
    'robotics_arxiv':'https://arxiv.org/pdf/1908.07332',
    'sensors_archive':'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7014400/fullTextXML',
    'jcmsi_pdf':'https://www.jstage.jst.go.jp/article/jcmsi/5/4/5_4_233/_pdf',
    'eventcamera_pdf':'https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Gossard_Table_Tennis_Ball_Spin_Estimation_with_an_Event_Camera_CVPRW_2024_paper.pdf',
    's2_10572328':'https://api.semanticscholar.org/graph/v1/paper/URL:https%3A%2F%2Fieeexplore.ieee.org%2Fabstract%2Fdocument%2F10572328?fields=title,year,externalIds,openAccessPdf',
}
def fetch(item):
    name,url=item
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=30) as response:
            data=response.read()
            ext='.pdf' if data.startswith(b'%PDF') else '.json' if 'json' in response.headers.get('Content-Type','') else '.html'
            root.joinpath(name+ext).write_bytes(data)
            details={'id':name,'url':url,'ok':True,'bytes':len(data),'file':name+ext}
            if ext=='.json':
                obj=json.loads(data); obj=obj.get('message',obj)
                details.update({k:obj.get(k) for k in ('title','DOI','published','abstract','link')})
            else:
                details['title']=re.findall(r'<title[^>]*>(.*?)</title>',data.decode('utf8','replace'),re.S)[:1]
            return details
    except Exception as exc: return {'id':name,'url':url,'ok':False,'error':str(exc)}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    output=list(pool.map(fetch,targets.items()))
root.joinpath('source_fetch_manifest.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf8')
for item in output: print(json.dumps(item,ensure_ascii=True))
