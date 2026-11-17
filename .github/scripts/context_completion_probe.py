"""Experimental completed-context proof, not admission-only or a standard gate.
Run only on the authorized DevLab server. Records complete raw responses and
memory/status history; requires genuine cache hits and32+ completed tokens.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
requests=None  # Loaded only for real HTTP probes; metadata/monitor and CPU tests need stdlib.
import signal
import re


def monitor_vram(output):
    """Caller starts before serve_alone, covering preparation and startup warmup."""
    done=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:done.set())
    signal.signal(signal.SIGINT,lambda *_:done.set())
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',buffering=1) as f:
        while not done.is_set():
            cards={}
            for used in Path('/sys/class/drm').glob('card[0-9]*/device/mem_info_vram_used'):
                dev=used.parent
                try:cards[str(dev.resolve())]={'used':int(used.read_text()),'total':int((dev/'mem_info_vram_total').read_text())}
                except OSError as e:cards[str(dev.resolve())]={'error':str(e)}
            f.write(json.dumps({'time':time.time(),'monotonic':time.monotonic(),'cards':cards})+'\n')
            done.wait(.25)


def check_sdk(root,pkg):
    """Require the card's actual wheel pin, including a nonmissing hash."""
    root,pkg=Path(root),Path(pkg)
    have=json.loads((root/'loom-install.json').read_text())
    want=json.loads((pkg/'llm_config.json').read_text()).get('sdk') or {}
    keys=('hrx_loom','source','wheel_sha256')
    ok=all(want.get(k) and have.get(k)==want[k] for k in keys)
    return {'passed':ok,'sdk_root':str(root),'expected':want,'actual':have,
            'limitation':'PyPI install metadata may omit source SHA; the exact pinned wheel hash is required and checked.'}


class ProbeInterrupted(InterruptedError):
    pass


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('pkg');ap.add_argument('url');ap.add_argument('out')
    ap.add_argument('--scripts',required=True)
    ap.add_argument('--context',type=int,required=True)
    ap.add_argument('--image',action='store_true')
    ap.add_argument('--cold-first',action='store_true',help='First successful generation is this near-cap request; retry only HTTP503 in the same server for at most300s, retaining every attempt.')
    ap.add_argument('--pin-primary',action='store_true')
    ap.add_argument('--concurrent',action='store_true')
    ap.add_argument('--timeout',type=float,default=120)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    global requests
    if requests is None:
        import requests as requests_module
        requests=requests_module
    sys.path.insert(0,a.scripts)
    from bench_common import Package,doc
    P=Package(a.pkg)
    marker='CAPACITY-CEDAR-731'
    text_question=f'Write six short factual sentences about photosynthesis. Then print {marker} as the final line.'
    question=('Describe the colored shapes and transcribe the image text. ' if a.image else '')+text_question
    thinking=False if P.chat.get('thinking')=='toggle' else None
    extra={'enable_thinking':False,'chat_template_kwargs':{'enable_thinking':False}}
    prefix='This is a capacity check. The repeated record is reference material. Follow the final question.\n'
    tail='\nFinal question: '+question
    overhead=len(P.ids(P.render_messages([{'role':'user','content':prefix+tail}],None,thinking)))
    budget=a.context-1024-overhead
    assert budget>1024
    ids=P.ids(doc(32000))
    while len(ids)<budget:ids+=ids
    filler=P.tok.decode(ids[:budget])
    text=prefix+filler+tail
    image_url=None
    if a.image:
        import vision_golden as V
        image_url=V.data_url(V.builtin_images()['shapes'][0])
    def body(text,image=False):
        content=[{'type':'image_url','image_url':{'url':image_url}},{'type':'text','text':text}] if image else text
        return {'model':P.model_id,'messages':[{'role':'user','content':content}],'temperature':0,'max_tokens':384,**extra}
    stop=threading.Event();samples=[]
    # One32GB AMD GPU is the campaign contract. Never count telemetry from an
    # unrelated iGPU as evidence of R9700 memory headroom; ambiguity fails.
    vram=[]
    for used in Path('/sys/class/drm').glob('card[0-9]*/device/mem_info_vram_used'):
        try:
            if (used.parent/'vendor').read_text().strip()=='0x1002' and int((used.parent/'mem_info_vram_total').read_text())>=30*2**30:vram.append(used)
        except OSError:pass
    if len(vram)!=1:raise RuntimeError(f'expected one32GB AMD target VRAM counter, found {len(vram)}')
    target_pci=str(vram[0].parent.resolve())
    target_total_bytes=int((vram[0].parent/'mem_info_vram_total').read_text())
    sample_lock=threading.Lock()
    sample_file=out/'memory.jsonl'
    sample_file.write_text('')
    def snapshot():
        d={'monotonic':time.monotonic(),'time':time.time()}
        try:
            r=requests.get(a.url+'/v1/status',timeout=2);d['status_http']=r.status_code;d['status']=r.json()
        except Exception as e:d['status_error']=str(e)
        d['vram_bytes']={}
        for path in vram:
            try:d['vram_bytes'][str(path.resolve())]=int(path.read_text())
            except OSError as e:d.setdefault('vram_errors',[]).append(str(e))
        with sample_lock:
            samples.append(d)
            with sample_file.open('a') as stream:stream.write(json.dumps(d)+'\n')
    def sample():
        while not stop.is_set():snapshot();stop.wait(0.25)
    snapshot()  # Durable initial telemetry exists before any request/control.
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    controls=[];responses=[];failure=None
    def post(tag,payload):
        before=time.monotonic();attempts=[]
        (out/(tag+'-request.json')).write_text(json.dumps({'time':time.time(),'payload':payload,'request_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()},indent=2)+'\n')
        while True:
            try:
                r=requests.post(a.url+'/v1/chat/completions',json=payload,timeout=a.timeout)
            except Exception as error:
                attempt={'elapsed_s':time.monotonic()-before,'time':time.time(),'status':0,'error_type':type(error).__name__,'error':str(error)}
                attempts.append(attempt)
                (out/(tag+'-attempts.json')).write_text(json.dumps(attempts,indent=2)+'\n')
                item={'tag':tag,**attempt,'response':{},'request_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()}
                (out/(tag+'.json')).write_text(json.dumps(item,indent=2)+'\n');responses.append(item)
                if isinstance(error,ProbeInterrupted):raise
                return item
            attempt={'elapsed_s':time.monotonic()-before,'time':time.time(),'status':r.status_code}
            if r.status_code!=200:attempt['body']=r.text
            attempts.append(attempt)
            (out/(tag+'-attempts.json')).write_text(json.dumps(attempts,indent=2)+'\n')
            if not (a.cold_first and tag=='near_first' and r.status_code==503 and time.monotonic()-before<300):break
            # This identical near-cap workload triggers asynchronous cold load.
            # Never restart the process or run a short successful warmup prompt.
            time.sleep(.5)
        try:d=r.json()
        except ValueError:d={'raw':r.text}
        item={'tag':tag,'elapsed_s':time.monotonic()-before,'status':r.status_code,'response':d,'request_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()}
        (out/(tag+'.json')).write_text(json.dumps(item,indent=2)+'\n');responses.append(item);return item
    def interrupt(*_):
        raise ProbeInterrupted('context probe terminated by SIGTERM')
    prior_sigterm=signal.signal(signal.SIGTERM,interrupt)
    try:
        if a.pin_primary:
            for endpoint,component in [('pin','primary'),('unpin','image')]:
                r=requests.post(a.url+'/v1/models/'+endpoint,json={'model':P.model_id,'component':component},timeout=30)
                controls.append({'action':endpoint,'component':component,'status':r.status_code,'body':r.text})
                r.raise_for_status()
        first=post('near_first',body(text,a.image))
        second=post('near_repeat',body(text,a.image))
        if a.concurrent:
            # Combined requested text fits the one pooled capacity, including
            # four output budgets and image/header allowance. This is c4 proof.
            long_budget=max(1024,budget-3*2048-1024)
            texts=[prefix+P.tok.decode(ids[:long_budget])+tail]+[prefix+P.tok.decode(ids[:1536])+'\nFinal question: '+text_question]*3
            barrier=threading.Barrier(4)
            def simultaneous(i,t):
                barrier.wait(timeout=30)
                return post(f'c4_{i}',body(t,a.image and i==0))
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(simultaneous,i,t) for i,t in enumerate(texts)]
                for f in futures:f.result()
    except Exception as error:
        failure={'error_type':type(error).__name__,'error':str(error),'time':time.time()}
        (out/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
    finally:
        stop.set();thread.join(timeout=3);snapshot()
        signal.signal(signal.SIGTERM,prior_sigterm)
        (out/'controls.json').write_text(json.dumps(controls,indent=2)+'\n')
    checks=[];answers=[]
    for r in responses:
        d=r['response'];choices=d.get('choices') or [];c=choices[0] if choices else {};message=c.get('message') or {};answer=message.get('content') or '';usage=d.get('usage') or {}
        completed=r['status']==200 and c.get('finish_reason')=='stop' and usage.get('completion_tokens',0)>=32
        words=set(re.findall(r'[a-z]+',answer.lower()))
        final_line=answer.rstrip().splitlines()[-1].strip('`* ') if answer.strip() else ''
        quality=final_line==marker and len(words & {'plants','plant','sunlight','light','chlorophyll','carbon','oxygen','sugar','glucose'})>=2
        if a.image and r['tag'] in ('near_first','near_repeat','c4_0'):
            quality=quality and V.about_shapes(answer) and 'LOOM' in answer.upper() and '42' in answer
        cached=(usage.get('prompt_tokens_details') or {}).get('cached_tokens',0)
        near=r['tag'] in ('near_first','near_repeat')
        actual=usage.get('prompt_tokens',0)
        checks.append({'tag':r['tag'],'completed':completed,'quality':quality,'actual_prompt_tokens':actual,'cached_tokens':cached,'near_context':not near or a.context-2048<=actual<a.context})
        if near:answers.append(answer)
    repeat=next((c for c in checks if c['tag']=='near_repeat'),{})
    def active_counts(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k in ('active_requests','active_rows') and isinstance(v,int):yield v
                yield from active_counts(v)
        elif isinstance(value,list):
            for v in value:yield from active_counts(v)
    observed_active=max((n for sample in samples for n in active_counts(sample.get('status'))),default=0)
    memory_observed=any(x.get('status_http')==200 and x.get('vram_bytes') for x in samples)
    request_sampled_peak_bytes=max((n for sample in samples for n in sample.get('vram_bytes',{}).values()),default=0)
    overlap_ok=not a.concurrent or observed_active>=4
    expected={'near_first','near_repeat'} | ({f'c4_{i}' for i in range(4)} if a.concurrent else set())
    ok=failure is None and {c['tag'] for c in checks}==expected and memory_observed and overlap_ok and all(c['completed'] and c['quality'] and c['near_context'] for c in checks) and len(answers)==2 and answers[0]==answers[1] and repeat.get('cached_tokens',0)>0
    result={'passed':ok,'failure':failure,'context':a.context,'image':a.image,'concurrent':a.concurrent,'checks':checks,'near_repeat_exact':len(answers)==2 and answers[0]==answers[1],'memory_observed':memory_observed,'target_pci':target_pci,'target_total_bytes':target_total_bytes,'request_sampled_peak_bytes':request_sampled_peak_bytes,'request_sampled_min_headroom_bytes':target_total_bytes-request_sampled_peak_bytes,'observed_max_active':observed_active,'c4_overlap_proved':overlap_ok,'cold_first':a.cold_first,'limitations':'Light semantic capacity proof; standard numerical/vision gates and throughput comparisons are separate. Startup warmup behavior belongs to the caller and must be disclosed.'}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return 0 if ok else 1


if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='--check-sdk':
        result=check_sdk(sys.argv[2],sys.argv[3]);print(json.dumps(result));sys.exit(0 if result['passed'] else 1)
    elif len(sys.argv)==3 and sys.argv[1]=='--monitor-vram':monitor_vram(sys.argv[2])
    else:sys.exit(main())
