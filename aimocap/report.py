"""Scientific quality report (HTML, Arabic, self-contained figures)."""
import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .skeleton import NAMES, FOOT, BONE_IDX, BONES

plt.rcParams.update({'figure.facecolor': 'white', 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9})
BLUE, ORANGE, GRAY, RED = '#2a6fdb', '#e8871e', '#8a98a8', '#d64545'


def _img(fig):
    b = io.BytesIO(); fig.savefig(b, format='png', dpi=110, bbox_inches='tight'); plt.close(fig)
    return '<img src="data:image/png;base64,' + base64.b64encode(b.getvalue()).decode() + '">'


def _figs(res):
    out = []
    fps = res['fps']; P = res['final']; T = len(P); t = np.arange(T) / fps
    # 1 quality per joint
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.bar(range(len(NAMES)), res['quality'].mean(0), color=BLUE)
    ax.set_xticks(range(len(NAMES))); ax.set_xticklabels(NAMES, rotation=70, fontsize=7)
    ax.set_ylim(0, 1); ax.set_title('Mean 3D quality per joint (confidence x views x reprojection)')
    out.append(('جودة كل مفصل', _img(fig)))
    # 2 reprojection per view over time
    fig, ax = plt.subplots(figsize=(9, 2.6))
    e = np.where(res['used'], res['err'], np.nan)
    with np.errstate(all='ignore'):
        for v in range(e.shape[0]):
            ax.plot(t, np.nanmedian(e[v], -1), lw=1, label=f'view {v}')
    ax.set_ylabel('reproj err (x panel h)'); ax.set_xlabel('s'); ax.legend(ncol=6, fontsize=7)
    ax.set_title('Reprojection error per view (spikes = AI-inconsistent frames)')
    out.append(('خطأ إعادة الإسقاط لكل زاوية', _img(fig)))
    # 3 bone length stability
    fig, ax = plt.subplots(figsize=(9, 2.8))
    def cv(X):
        bl = np.linalg.norm(X[:, BONE_IDX[:, 1]] - X[:, BONE_IDX[:, 0]], axis=-1)
        return np.nanstd(bl, 0) / (np.nanmean(bl, 0) + 1e-9) * 100
    x = np.arange(len(BONES))
    ax.bar(x - .27, cv(res['raw']), .27, color=GRAY, label='raw')
    ax.bar(x, cv(res['triangulated']), .27, color=ORANGE, label='robust triangulation')
    ax.bar(x + .27, cv(res['final']), .27, color=BLUE, label='final')
    ax.set_xticks(x); ax.set_xticklabels([f'{a[:8]}-{b[:8]}' for a, b in BONES], rotation=75, fontsize=6)
    ax.set_ylabel('bone length CV %'); ax.legend(fontsize=7); ax.set_yscale('symlog', linthresh=1)
    ax.set_title('Bone length variability (lower = anatomically consistent)')
    out.append(('ثبات أطوال العظام', _img(fig)))
    # 4 feet
    fig, ax = plt.subplots(figsize=(9, 2.6))
    for s, c in (('left', ORANGE), ('right', BLUE)):
        h = P[:, FOOT[s], 1].min(1)
        ax.plot(t, h, color=c, lw=1.2, label=f'{s} foot height')
        m = res['contacts'][s]
        ax.fill_between(t, 0, np.where(m, h.max() * .15, 0), color=c, alpha=.18, step='mid')
    ax.set_ylabel('m'); ax.set_xlabel('s'); ax.legend(fontsize=7)
    ax.set_title('Foot height and detected ground contacts (shaded)')
    out.append(('القدمان والتلامس مع الأرض', _img(fig)))
    # 5 angles
    fig, ax = plt.subplots(figsize=(9, 2.6))
    A = res['angles']
    for k, c in (('knee_L', ORANGE), ('knee_R', BLUE), ('elbow_L', '#f4b26a'), ('elbow_R', '#8db3f0')):
        ax.plot(t, A[k], color=c, lw=1.1, label=k)
    ax.set_ylabel('deg'); ax.set_xlabel('s'); ax.legend(ncol=4, fontsize=7); ax.set_title('Joint angles (180 = straight)')
    out.append(('زوايا المفاصل', _img(fig)))
    # 6 COM
    fig, ax = plt.subplots(1, 2, figsize=(9, 2.8))
    com = res['com']
    ax[0].plot(com[:, 0], com[:, 2], color=BLUE, lw=1.2); ax[0].set_aspect('equal', 'datalim')
    ax[0].set_xlabel('x (m)'); ax[0].set_ylabel('z (m)'); ax[0].set_title('COM path (top view)')
    ax[1].plot(t, com[:, 1], color=BLUE, lw=1.2); ax[1].set_xlabel('s'); ax[1].set_ylabel('m'); ax[1].set_title('COM height')
    out.append(('مركز الثقل (نموذج de Leva)', _img(fig)))
    return out


def write_report(res, path):
    r = res['report']; m = r['metrics']
    cams = res['cameras']
    def row(k, lab, unit, better='low'):
        a, b, c = m['raw'].get(k), m['triangulated'].get(k), m['final'].get(k)
        f = lambda v: '—' if v is None else f'{v:.3f}'
        return f'<tr><td>{lab}</td><td>{f(a)}</td><td>{f(b)}</td><td><b>{f(c)}</b></td><td>{unit}</td></tr>'
    camrows = ''
    C = cams.centers()
    for v in range(cams.V):
        d = np.linalg.norm(C[v, [0, 2]]); yaw = np.degrees(np.arctan2(C[v, 0], C[v, 2])) % 360
        e = r['reproj_err_median_per_view'][v]
        camrows += (f'<tr><td>{v}</td><td>{cams.yaws[v]:.0f}°</td><td>{yaw:.1f}°</td><td>{d:.2f}</td>'
                    f'<td>{C[v,1]:.2f}</td><td>{cams.f[v]:.0f}</td><td>{"—" if e is None else f"{e:.4f}"}</td></tr>')
    sw = ''.join(f'<tr><td>{g}</td><td>{", ".join(str(x) for x in v)}</td></tr>' for g, v in r['swaps'].items())
    figs = ''.join(f'<section><h2>{h}</h2>{img}</section>' for h, img in _figs(res))
    html = f'''<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8"><title>تقرير جودة الحركة</title>
<style>body{{font:14px Tahoma,Segoe UI,Arial;max-width:1000px;margin:24px auto;padding:0 16px;color:#1b2430;background:#fff}}
h1{{font-size:22px}}h2{{font-size:16px;margin-top:28px;border-bottom:1px solid #e3e8ee;padding-bottom:6px}}
table{{border-collapse:collapse;width:100%;margin:8px 0}}td,th{{border:1px solid #e3e8ee;padding:6px 8px;text-align:center}}
th{{background:#f4f7fa}}img{{max-width:100%;direction:ltr}}.k{{display:flex;gap:12px;flex-wrap:wrap}}
.k div{{background:#f4f7fa;border-radius:10px;padding:10px 14px;min-width:150px}}.k b{{display:block;font-size:20px}}</style></head><body>
<h1>تقرير جودة التقاط الحركة من فيديو AI</h1>
<div class="k"><div>الإطارات<b>{r["frames"]}</b></div><div>الزوايا<b>{r["views"]}</b></div><div>FPS<b>{r["fps"]:.1f}</b></div>
<div>متوسط الزوايا المستخدمة<b>{r["views_used_mean"]:.2f}</b></div><div>عينات مُكمَلة<b>{r["filled_samples_pct"]:.1f}%</b></div><div>زمن المعالجة<b>{r["seconds"]}s</b></div></div>
<h2>المقاييس: خام ← تثليث قوي ← نهائي</h2>
<table><tr><th>المقياس</th><th>خام</th><th>تثليث قوي</th><th>نهائي</th><th>الوحدة</th></tr>
{row("bone_length_cv_pct","تذبذب أطوال العظام (CV)","%")}{row("mean_accel_m_s2","الارتعاش (متوسط التسارع)","m/s²")}
{row("foot_skate_m_s","انزلاق القدم أثناء التلامس","m/s")}{row("ground_penetration_frames","إطارات اختراق الأرض","إطار")}</table>
<h2>التصحيحات الذكية المطبَّقة</h2><table>
<tr><td>نقاط ثنائية الأبعاد مرفوضة كأخطاء AI (قفزات/تشوّه)</td><td>{r["glitch_keypoints"]}</td></tr>
<tr><td>إطارات زوايا كاملة مرفوضة (تشوّه شامل)</td><td>{r["glitch_view_frames"]}</td></tr>
<tr><td>إطارات زوايا غير متسقة هندسياً (هلوسة زاوية)</td><td>{r["inconsistent_view_frames"]}</td></tr>
<tr><td>قفزات ثلاثية الأبعاد أزيلت (Hampel)</td><td>{r["spikes_removed"]}</td></tr>
<tr><td>فرط انبساط الركبة المصحَّح</td><td>{r["knee_hyperextension_fixed"]}</td></tr>
<tr><td>عكس كامل لليمين/اليسار</td><td>{"نعم" if r["global_mirror_fixed"] else "لا"}</td></tr>
<tr><td>تلامس القدم (يسار / يمين)</td><td>{r["contact_pct"]["left"]:.0f}% / {r["contact_pct"]["right"]:.0f}%</td></tr></table>
<h3>تبديل يمين/يسار المصحَّح (عدد الإطارات لكل زاوية)</h3><table><tr><th>المجموعة</th><th>الزوايا 0..N</th></tr>{sw}</table>
<h2>الكاميرات الافتراضية بعد المعايرة الذاتية</h2>
<table><tr><th>زاوية</th><th>الاتجاه المفترض</th><th>الاتجاه المقدَّر</th><th>المسافة (م)</th><th>الارتفاع (م)</th><th>البعد البؤري (px)</th><th>خطأ الإسقاط الوسيط</th></tr>{camrows}</table>
<p style="color:#5b6b7b">المعايرة تتم بضبط الحزم (bundle adjustment) باستخدام نقاط جسم الشخص كهدف معايرة، مع تثبيت الزاوية 0 وتنظيم ضعيف نحو التوزيع الاسمي. المسافات بوحدة نسبية قبل التحجيم بطول الشخص.</p>
{figs}
<h2>المنهجية</h2><ol>
<li>كشف 26 نقطة (RTMPose Halpe-26 مع القدمين) في كل لوحة.</li>
<li>فحص أخطاء AI ثنائية الأبعاد: انحراف عن الوسيط الزمني بأكثر من 6×MAD.</li>
<li>معايرة ذاتية: اختيار ترتيب الزوايا تلقائياً ثم ضبط الحزم بخسارة soft-L1.</li>
<li>حل تبديل يمين/يسار لكل مجموعة أطراف بخوارزمية Viterbi (خطأ الإسقاط + كلفة التبديل الزمني).</li>
<li>تثليث DLT موزون بالثقة مع إسقاط الزاوية الأسوأ تكرارياً، ورفض الزوايا غير المتسقة.</li>
<li>إكمال الفجوات (PCHIP)، مرشّح Hampel، مرشّح Butterworth صفري الطور (4th order).</li>
<li>مستوى الأرض بـ RANSAC، وتحجيم بطول الشخص، وتوجيه نحو +Z.</li>
<li>قيود أطوال العظام (PBD) مع تناظر يمين/يسار، ومنع فرط انبساط الركبة.</li>
<li>كشف تلامس القدم، وتثبيتها ضد الانزلاق، ومنع اختراق الأرض، ومركز الثقل بنموذج de Leva (1996).</li></ol>
</body></html>'''
    open(path, 'w', encoding='utf-8').write(html)
