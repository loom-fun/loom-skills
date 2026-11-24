"""Alternative exact schedules for the staged serial FP32 recurrence."""
import re

def wave_interleave(source, width=4):
    lines=source.splitlines(keepends=True)
    for first, last, family in [("p0","n3","p"),("o0","out3","o")]:
        start=next(i for i,l in enumerate(lines) if l.lstrip().startswith(f"%{first} = vector.dotf"))
        stop=next(i for i,l in enumerate(lines[start:], start) if l.lstrip().startswith(f"%{last} = ")) + 1
        blocks=[]
        for j in range(4):
            a=next(i for i,l in enumerate(lines[start:stop],start) if l.lstrip().startswith(f"%{family}{j} = vector.dotf"))
            if j: blocks[-1]=lines[begin:a]
            begin=a
            blocks.append([])
        blocks[-1]=lines[begin:stop]
        assert len(set(map(len,blocks)))==1
        ordered = []
        for k in range(0,4,width):
            for group in zip(*blocks[k:k+width]):
                ordered.extend(group)
        lines[start:stop]=ordered
    return "".join(lines)

def precalc_body(source):
    start = source.index("    %alpha_biased =")
    end = source.index("    %qscale =",start)
    return source[:start]+"""    %decay_vector = vector.splat %al : vector<16xf32>
    %kdecay = scalar.mulf %ki, %al : f32
"""+source[end:].replace("%beta_value", "%be")

precompute = """
kernel.def export("prepare_factors") @prepare_factors(%t: index) {
  %one = index.constant 1 : index
  %lanes = index.constant 32 : index
  kernel.launch.config workgroups(%t, %one, %one) workgroup_size(%lanes, %one, %one) : index
} launch(%tokens_arg: index, %alpha: buffer, %beta: buffer, %a: buffer, %dt: buffer, %alpha_o: buffer, %beta_o: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %t0 = kernel.workgroup.id<x> : index
  %t = index.assume %t0 [lt(%t0, %tokens)] : index
  %h0 = kernel.workitem.id<x> : index
  %h = index.assume %h0 [range(%h0, 0, 31)] : index
  %z = index.constant 0 : offset
  %av = buffer.view %alpha[%z] : buffer -> view<[%tokens]x32xf32>
  %bv = buffer.view %beta[%z] : buffer -> view<[%tokens]x32xf32>
  %as = buffer.view %a[%z] : buffer -> view<32xf32>
  %dv = buffer.view %dt[%z] : buffer -> view<32xf32>
  %ao = buffer.view %alpha_o[%z] : buffer -> view<[%tokens]x32xf32>
  %bo = buffer.view %beta_o[%z] : buffer -> view<[%tokens]x32xf32>
  %al = view.load %av[%t, %h] : view<[%tokens]x32xf32> -> f32
  %be = view.load %bv[%t, %h] : view<[%tokens]x32xf32> -> f32
  %ssma = view.load %as[%h] : view<32xf32> -> f32
  %bias = view.load %dv[%h] : view<32xf32> -> f32
  %biased = scalar.addf %al, %bias : f32
  %softplus = scalar.softplusf<afn> %biased : f32
  %gate = scalar.mulf %softplus, %ssma : f32
  %decay = scalar.expf<afn> %gate : f32
  %prob = scalar.logisticf<afn> %be : f32
  view.store %decay, %ao[%t, %h] : f32, view<[%tokens]x32xf32>
  view.store %prob, %bo[%t, %h] : f32, view<[%tokens]x32xf32>
  kernel.return
}
"""

def add_precompute_cases(cases):
    def add(m):
        launch=m.group()
        t = re.search(r"tensor<(\d+)x32xf32>", launch)[1]
        ty=f"tensor<{t}x32xf32>"
        return (f"  %pa = check.generate.fill value(0.0) : {ty}\n  %pb = check.generate.fill value(0.0) : {ty}\n"
                f"  kernel.launch @prepare_factors[%t](%t, %alpha, %beta, %a, %dt, %pa, %pb) : [index](index, {ty}, {ty}, tensor<32xf32>, tensor<32xf32>, {ty}, {ty})\n"
                +launch.replace("%alpha, %beta, %a, %dt","%pa, %pb, %a, %dt"))
    return re.sub(r"^  kernel\.launch @g_gdn_recurrence_pf\[.*$", add, cases, flags=re.M)
