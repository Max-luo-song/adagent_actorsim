

def infer_five(model, namespace):
    from uniad_single import TAIL
    old = "_cameras = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT',\n            'CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']"
    if TAIL.count(old)!=1:
        raise ValueError('single-inference camera declaration differs')
    changes = adapt_camera_modules(model)
    print('Camera adaptations:',json.dumps(changes),flush=True)
    tail = TAIL.replace(old,'_cameras = '+repr(ORDER)).replace(
        'virtual_rear_views=True','virtual_rear_views=False')
    scope = dict(namespace,model=model)
    exec(compile(tail,'five_camera_inference','exec'),scope)
    report = scope['_report']
    report.update(status='five real-camera UniAD inference passed',
                  pretrained_camera_count=6,inference_camera_count=5,
                  camera_adaptations=changes,
                  side_embedding_initialization='pretrained back-left and back-right rows',
                  trained_or_finetuned_on_waymo=False)
    out = Path(os.environ['UNIAD_SMOKE_OUTPUT'])
    (out/'uniad_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


def infer():
    base = Path(os.environ['BASE']);job = Path(os.environ['FIVE_JOB'])
    packet = job/'observation.pkl'
    repo = base/'code/UniAD_SIM';source = repo/'tools/closeloop/e2e.py'
    if not packet.is_file():
        raise FileNotFoundError('run --prepare first: '+str(packet))
    out = job/'prediction';out.mkdir(parents=True,exist_ok=True)
    os.environ['UNIAD_PACKET'] = str(packet)
    os.environ['UNIAD_SMOKE_OUTPUT'] = str(out)
    sys.path[:0] = [str(repo),str(source.parent)]
    os.chdir(repo)
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body
              if isinstance(n,ast.FunctionDef) and n.name=='main')
    cuts = [i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='cameras' for t in n.targets)]
    if len(cuts)!=1:
        raise ValueError('unexpected UniAD initializer structure')
    fn.body = fn.body[:cuts[0]]+ast.parse(
        'from five_uniad import infer_five as _infer_five\n_infer_five(model, globals())').body
    sys.argv = [str(source),'projects/configs/stage2_e2e/base_e2e.py',
                'ckpts/uniad_base_e2e.pth','--launcher','none','--eval','bbox',
                '--tmpdir',str(out/'tmp'),'--output',str(out)]
    exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),
         {'__name__':'__main__','__file__':str(source)})


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        infer()
