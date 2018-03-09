import collections
import matplotlib.pyplot as plt
import numpy as np
import analytics.utils as utils

__all__ = [
    "count_part_freq",
    "fill_out_histogram",
    "normalize_histogram",
    "plot_hist"]


def rec_normalize_summary_histogram(lvl, hist):
    import numbers
    '''
    For each level in a dictionary, convert all integers to their
    proportion of the sum of all integers AT THAT LEVEL.
    '''
    total = 0.0
    for k, dat in hist.items():
        if isinstance(dat, dict):
            hist[k] = rec_normalize_summary_histogram(lvl + 1, dat)
        elif k == 'total':
            continue
        elif isinstance(dat, numbers.Number):
            total += dat
        else:
            print(
                "Type {} currently not normalizeable at level {}.".format(
                    dat, lvl))
    for k, dat in hist.items():
        if isinstance(dat, numbers.Number):
            hist[k] = float(dat) / total
    return hist


def normalize_general_histogram(hist):
    import copy
    hist_ = copy.deepcopy(hist)
    return rec_normalize_summary_histogram(0, hist_)


def summarize_histogram(hist, combine=False):
    keys = ['obj', 'part', 'total']
    if combine:
        keys.extend(['ambiguous'])
    else:
        keys.extend(['imp_to_tell', 'other'])
    overview = {k: 0 for k in keys}
    summary = {}
    for obj_cls, obj_data in hist.items():
        clsid = int(obj_cls)
        summary[clsid] = collections.defaultdict(int)
        for part, freq in obj_data['obj'].items():
            summary[clsid]['obj'] += freq
        for part, freq in obj_data['part'].items():
            summary[clsid]['part'] += freq
        for part, freq in obj_data['other'].items():
            summary[clsid]['other'] += freq
        for part, freq in obj_data['imp_to_tell'].items():
            summary[clsid]['imp_to_tell'] += freq
        # Optionally treat "impossible to tell" and "other" as one
        if combine:
            summary[clsid]['ambiguous'] = summary[clsid].pop(
                'imp_to_tell', None) + summary[clsid].pop('other', None)

        total = 0
        for cat, freq in summary[clsid].items():
            total += freq
            overview[cat] += freq
        summary[clsid]['total'] = total
        overview['total'] += total
        summary[clsid] = dict(summary[clsid])
    return {'overview': overview, 'by_class': summary}


def count_key_freq(rb_key):
    import copy
    rb_key_ = copy.deepcopy(rb_key)
    for k, responses in rb_key_.items():
        max_freq = 0
        max_cls = None
        M = float(len(responses))
        choices = collections.defaultdict(int)
        for response in responses:
            answer = response['answer']
            choices[answer] += 1

        for cls, freq in choices.items():
            if freq > max_freq:
                max_freq = freq
                max_cls = cls
        # max_clses = set()
        # for cls, freq in responses.items():
        #     if freq == max_freq:
        #         max_clses.add(cls)
        if max_freq <= M / 2.0:  # If it's an split, "impossible to tell"
            max_cls = -3
        rb_key_[k] = max_cls
    return rb_key_


def get_distribution_with_consensus(rb_key):
    keys = ['obj', 'part', 'imp_to_tell', 'ambiguous', 'other']
    overall = {k: 0 for k in keys}
    by_class = {}
    for k, ans in rb_key.items():
        keydat = utils.separate_key(k)
        # Initialize for object if new
        if keydat['obj_id'] not in by_class:
            by_class[keydat['obj_id']] = {k: 0 for k in keys}

        if ans == keydat['obj_id']:
            by_class[keydat['obj_id']]['obj'] += 1
            overall['obj'] += 1
        elif ans == keydat['part_id']:
            by_class[keydat['obj_id']]['part'] += 1
            overall['part'] += 1
        elif ans == "-1":
            by_class[keydat['obj_id']]['imp_to_tell'] += 1
            overall['imp_to_tell'] += 1
        elif ans == "-2":
            by_class[keydat['obj_id']]['other'] += 1
            overall['other'] += 1
        elif ans == -3:
            by_class[keydat['obj_id']]['ambiguous'] += 1
            overall['ambiguous'] += 1
        else:
            raise RuntimeError("Type {}: {} not found".format(type(ans), ans))
    return {'overall': overall, 'by_class': by_class}


def count_part_freq(rbo, dInfo):
    histogram = {}
    for obj, responses in rbo.items():
        objhist = {
            'obj': collections.defaultdict(int),
            'part': collections.defaultdict(int),
            'other': collections.defaultdict(int),
            'imp_to_tell': collections.defaultdict(int)}
        for resp in responses:
            for out in resp['output']:
                for k, v in out.items():
                    nearest_part = dInfo[k]['nearest_part_id']
                    ans = v['answer']
#                     if ans == '100':
#                         ans = nearest_part
                    if ans == obj:
                        objhist['obj'][nearest_part] += 1
                    elif int(ans) == -1:
                        objhist['imp_to_tell'][nearest_part] += 1
                    elif ans == "NA" or int(ans) == -2:
                        objhist['other'][nearest_part] += 1
                    else:
                        objhist['part'][nearest_part] += 1
        histogram[obj] = objhist
    return histogram


def normalize_histogram(hist):
    for obj, v in hist.items():
        ref_obj = v['obj']
        ref_part = v['part']
        ref_imp_to_tell = v['imp_to_tell']
        # ref_other =  v['other']
        allkeys = list(set(ref_obj.keys()) | set(ref_part.keys()))
        frequencies = {}
        for k in allkeys:
            freq_obj_ref = float(ref_obj[k]) if k in ref_obj else 0.0
            freq_part_ref = float(ref_part[k]) if k in ref_part else 0.0
            freq_imp_to_tell_ref = float(
                ref_imp_to_tell[k]) if k in ref_imp_to_tell else 0.0
            # May wish to remove the imp_to_tell reference in the denominator
            denom = freq_obj_ref + freq_part_ref + freq_imp_to_tell_ref
            if denom == 0.0:
                val_obj = 0.0
                val_part = 0.0
            else:
                val_obj = freq_obj_ref / denom
                val_part = freq_part_ref / denom
            frequencies[k] = {'obj': val_obj, 'part': val_part}
        hist[obj] = frequencies


def fill_out_histogram(hist, reference):
    '''Adds frequency of zero to all nodes in the histogram
       which were not annotated.'''
    import copy
    hist_ = copy.deepcopy(hist)
    for obj, v in reference.items():
        if obj not in hist_:
            continue
        for part in v['parts']:
            if int(part) not in hist_[obj]:
                hist_[obj][int(part)] = {'obj': 0.0, 'part': 0.0}
    return hist_


def plot_hist(hr_hist, component='obj'):
    num = len(hr_hist)
    f, axarr = plt.subplots(
        int(np.round(float(num) / 2.0 + 0.001)), 2, figsize=(10, 25))
    for i, (obj, part_data) in enumerate(hr_hist.items()):
        a = i // 2
        b = i % 2
        vals = [v[component] for v in part_data.values()]
        axarr[a, b].bar(range(len(part_data)), vals, align='center')
        # , list(part_data.keys()))
        axarr[a, b].set_xticks(range(len(part_data)))
        axarr[a, b].set_xticklabels(list(part_data.keys()), rotation=90)
        axarr[a, b].set_title("{}".format(obj))
        # plt.bar(range(len(part_data)), list(part_data.values()),
        # align='center')
        # plt.xticks(range(len(part_data)), list(part_data.keys())
    f.tight_layout()
    plt.show()


def _rec_plot_this_hist(title, hist):
    from PIL import Image
    vals = [v for k, v in hist.items() if k != 'total']
    labels = [k for k, v in hist.items() if k != 'total']
    num = len(vals)
    plt.figure()
    plt.bar(range(num), vals, align='center')
    plt.xticks(range(num), labels, rotation='vertical')
    plt.title(title)
    # plt.show()
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = Image.frombytes('RGB', canvas.get_width_height(),
                                canvas.tostring_rgb())
    return pil_image


def plot_histograms(title, hist):
    import numbers
    imgs = []
    for k, v in hist.items():
        if isinstance(v, dict) and np.alltrue(
                [isinstance(val, numbers.Number) for val in v.values()]):
            imgs.append(_rec_plot_this_hist(k, v))
        elif isinstance(v, dict):
            imgs.extend(plot_histograms(0, v))
    return imgs
