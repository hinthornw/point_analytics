'''
This module contains functions to clean raw data returned from Mturk, 
reorganize to be indexed by object, etc., as well as other functions.

'''
import collections
import numpy as np
import importlib
from skimage.morphology import medial_axis
from PIL import Image
importlib.import_module('.utils', 'point_analytics')
importlib.import_module('.visualization', 'point_analytics')

__all__ = [
    "point2annid",
    "get_valid_circle_indices",
    "collect_all_key_info",
    "collect_real_instances",
    "update_dInfo_instances",
    "get_all_real_parts",
    "responses_by_object"]


def na2numeric(responses, to='-1'):
    def convert_na(frame, to='-1'):
        keys = list(frame)
        for key in keys:
            ans = frame[key]['answer']
            if ans == 'NA':
                frame[key]['answer'] = str(to)
    f = utils.bake_function(convert_na, to=to)
    utils.map2each_answer(responses, f)


def get_keys_certain(
        answers_by_key,
        min_number=3,
        which="either",
        min_annotated=3,
        max_display=200):
    assert(which in ['either', 'both', 'part', 'obj'])
    uncertain_keys = {}
    for k, hist in answers_by_key.items():
        keydat = utils.separate_key(k)
        obj_id = int(keydat['obj_id'])
        part_id = int(keydat['part_id'])
        obj_ref = hist[obj_id] if obj_id in hist else 0
        part_ref = hist[part_id] if part_id in hist else 0
        na_ref = hist[-2] if -2 in hist else 0
        imp_to_tell_ref = hist[-1] if -1 in hist else 0
        total = obj_ref + part_ref + na_ref + imp_to_tell_ref
        if total < min_annotated:
            continue
        # Originally only 3 annotators per key. Messed that up.
        threshold = float(min_number) / 3.0 * total
        if which == 'part':
            if part_ref >= threshold:
                uncertain_keys[k] = hist
        elif which == 'obj' or which == 'object':
            if obj_ref >= threshold:
                uncertain_keys[k] = hist
        elif which == "either":
            if part_ref >= min_number or obj_ref >= threshold:
                uncertain_keys[k] = hist
        elif which == "both":
            if part_ref + obj_ref >= threshold:
                uncertain_keys[k] = hist
        if len(uncertain_keys) == max_display:
            break
    return uncertain_keys


def get_keys_uncertain(answers_by_key, min_number=1, which="both"):
    uncertain_keys = {}
    for k, hist in answers_by_key.items():
        impossible_to_tell = hist[-1] if -1 in hist else 0
        nan = hist[-2] if -2 in hist else 0
        if which == -1:
            if impossible_to_tell >= min_number:
                uncertain_keys[k] = hist
        elif which == -2:
            if nan >= min_number:
                uncertain_keys[k] = hist
        elif which == "both":
            if impossible_to_tell + nan >= min_number:
                uncertain_keys[k] = hist
    return uncertain_keys


def get_keys_thresholded(answers_by_key, threshold=0.6):
    key_to_answer = {}
    discord = {}
    assert(threshold >= 0 and threshold <= 1)
    for k, hist in answers_by_key.items():
        total = np.array(list(hist.values())).sum()
        mode = -2
        max_num = -1
        for answer, num in hist.items():
            if num > max_num and num > total * float(threshold):
                max_num = num
                mode = answer
        if max_num != -1:
            key_to_answer[k] = mode
        else:
            discord[k] = hist
    return key_to_answer, discord


def get_answers_by_key(rb_key):
    answers_by_key = {}
    for k, responses in rb_key.items():
        answers_by_key[k] = collections.defaultdict(int)
        for response in responses:
            ans = int(response['answer'])
            answers_by_key[k][ans] += 1
        answers_by_key[k] = dict(answers_by_key[k])
    return answers_by_key


# Assumes all elements in a response are the same..
def get_object(response, which='obj'):
    val = {'obj': 1, 'inst': 2, 'part': 3}
    key = list(response['output'][0])[0]
    elems = key.rsplit('_', 5)
    if which in val:
        return elems[val[which]]
    else:
        raise RuntimeError(
            "{} type not contained within response key. Valid values"
            "are 'obj', 'inst', and 'part'")


def key2annid(key, details):
    '''
    Given a key, find the cooresponding annotation id
    (or alternatively, instance id). Takes in the
    details API object and a key. Returns an annid.
    Note that error checking is hacky and only
    looks for points which are out of bounds.
    '''
    keydat = utils.separate_key(key)
    category_id = int(keydat['obj_id'])
    pid = int(keydat['part_id'])
    x, y = int(keydat['xCoord']), int(keydat['yCoord'])
    imid = int("".join(keydat['imid'].split("_")))
    annid = 0
    if pid == 100:
        mask = details.getMask(img=imid, cat=category_id)
        h, w = mask.shape
        annid = mask[y, x]
        if annid != 0:
            return int(annid)
        else:
            eyes, jays = get_valid_circle_indices(mask, (y, x), 1)
            annid = mask[eyes, jays]
            annid = [_id for _id in annid if _id > 0]
            if len(annid) != 0:
                return int(annid[0])
            eyes, jays = get_valid_circle_indices(mask, (y, x), 5)
            annid = mask[eyes, jays]
            annid = [_id for _id in annid if _id > 0]
            if len(annid) != 0:
                return int(annid[0])
            else:
                print(eyes, jays)
                print(mask[eyes, jays])
                # raise RuntimeWarning("Error for key {}. Point out of"
                #                      "instance mask bounds.".format(key))
                print("Error for key {}. Point out of"
                      "instance mask bounds.".format(key))

    anns = details.getAnns(imgs=imid, cats=category_id)
    for ann in anns:
        annid = 0
        for part in ann['parts']:
            if pid != 100:  # Search through all parts if it's silhouette
                if part['part_id'] != pid:
                    continue
            segmentation = part['segmentation']
            mask = details.decodeMask(segmentation)

            if mask[y, x] != 0:
                return int(ann['id'])
            else:
                eyes, jays = get_valid_circle_indices(mask, (y, x), 3)
                annid = mask[eyes, jays]
                annid = [_id for _id in annid if _id > 0]
                if len(annid) != 0:
                    # return int(annid[0])
                    return int(ann['id'])
                else:
                    continue

    if annid == 0:
        raise RuntimeError(
            "Error for key {}. Point out of instance mask bounds.".format(key))
    print(key)
    print(mask[eyes, jays])

    return int(annid)


def point2annid(question, details, D, dInfo, r=3, update=False):
    keys = list(question)
    for key in keys:
        if key in D:
            if not update:  # Toggle whether to recompute those already cached
                continue
        key_data = dInfo[key]
        x, y = int(key_data['xCoord']), int(key_data['yCoord'])
        imid = key_data['imid']
        imid = int(imid[0:4] + imid[5:])
        part_id = int(key_data['nearest_part_id'])
        gen_id = int(key_data['part_id'])
        anns = details.getAnns(imgs=[imid])
        D[key] = []
        for ann in anns:
            annid = ann['id']
            part_present = False
            for part in ann['parts']:
                if int(part['part_id']) == part_id:
                    mask = details.decodeMask(part['segmentation'])
                    eyes, jays = get_valid_circle_indices(
                        mask, (y, x), r)
                    if mask[eyes, jays].sum() > 0:
                        D[key].append(annid)
                        part_present = True
                        break
            if part_present:
                continue
            for part in ann['parts']:
                if int(part['part_id']) == gen_id:
                    mask = details.decodeMask(part['segmentation'])
                    eyes, jays = get_valid_circle_indices(
                        mask, (y, x), r)
                    if mask[eyes, jays].sum() > 0:
                        D[key].append(annid)
                        break

        # Choosing the annotation which places the point furthest from the mask
        # boundary
        if len(D[key]) <= 1:
            return
        maxDist = 0
        maxAnnid = 0
        anns = details.getAnns(annIds=D[key])
        for ann in anns:
            annid = ann['id']
            for part in ann['parts']:
                if int(part['part_id']) in [gen_id, part_id]:
                    mask = details.decodeMask(part['segmentation'])
                    med, distance = medial_axis(mask, return_distance=True)
                    if distance[y, x] > maxDist:
                        maxDist = distance[y, x]
                        maxAnnid = annid
        D[key] = [maxAnnid]


def responses_by_object(responses):
    data = collections.defaultdict(list)
    for response in responses:
        vals = utils.separate_key(list(response['output'][0])[0])
        obj = vals['obj_id']
        data[obj].append(response)
    return dict(data)


def responses_by_hit_id(responses):
    rb_hit_id = collections.defaultdict(list)
    for i, response in enumerate(responses):
        rb_hit_id[response['hit_id']].append(response['output'])
    return dict(rb_hit_id)


def responses_by_key(responses):
    rb_key = collections.defaultdict(list)
    for i, response in enumerate(responses):
        for j, frame in enumerate(response['output']):
            for key, data in frame.items():
                rb_key[key].append(data)
    return dict(rb_key)


def collect_info_from_keys(response, dInfo):
    output = response['output']
    for frame in output:
        for k in list(frame):
            dInfo[k] = utils.separate_key(k)


def collect_all_key_info(responses):
    dInfo = {}
    for r in responses:
        collect_info_from_keys(r, dInfo)
    return dInfo


def collect_real_instances(dInfo, details, cached):
    '''Instance IDs are currently stored as 0,1,etc.
       but have unique ids otherwise'''
    key2inst = {}
    for key, data in dInfo.items():
        if key in cached:
            key2inst[key] = cached[key]
            continue
        imid = data['imid']
        imid = int("".join(imid.split("_")))
        obj = data['obj_id']
        # inst = int(data['inst_id'])
        mask = details.getMask(img=imid, cat=int(obj))
        i, j = int(data['yCoord']), int(data['xCoord'])
        inst = int(mask[i, j])
        if inst == 0:
            print(key, inst)
            eyes, jays = get_valid_circle_indices(mask, (i, j), 2)
            candidates = np.array([int(cand) for cand in mask[eyes, jays]
                                   if int(cand) != 0])
            counts = np.bincount(candidates)
            inst = np.argmax(counts)
        key2inst[key] = inst  # int(mask[i, j])
    return key2inst


def update_dInfo_instances(dInfo, key2inst):
    for k, v in dInfo.items():
        v['inst_id'] = key2inst[k]


def collect_uncertain(rbhid):
    '''rbhid ~ responses by HIT id: dictionary is
        (HIT ID: [resp1, resp2, ...]) pairs'''
    for hid, responses in rbhid.items():
        pass


def key2real_part(dInfo, details, dInstances):
    ref = {}
    for k, v in dInfo.items():
        val = {acc: dat for acc, dat in v.items()}
        if val['part_id'] == '100':
            actual_part = silh2part(v, details)
            val['part_id'] = actual_part
        ref[k] = val
    return ref


# Not done yet TODO
def dist(center, r, h, w):
    i, j = center
    if r == 0:
        return ([i], [j])
    elif r < 0:
        raise RuntimeError("Radius of circle must be nonnegative.")
    # eyes = []
    # jays = []
    search_space = None  # np.ogrid()
    return search_space


def get_valid_circle_indices(arr, center, r):
    h, w = arr.shape
    i, j = center
    i = min(max(i, 0), h - 1)
    j = min(max(j, 0), w - 1)
    if r > 0:
        istart, istop = max(i - r, 0), min(i + r + 1, h)
        jstart, jstop = max(j - r, 0), min(j + r + 1, w)
        eyes = [y for y in range(istart, istop) for _ in range(jstart, jstop)]
        jays = [x for _ in range(istart, istop) for x in range(jstart, jstop)]
#         sqdists = [((y-i)**2+(x-j)**2, (y,x)) for y in
# range(istart,istop) for x in range(jstart, jstop)]
    else:
        eyes = [i]
        jays = [j]
    return eyes, jays


def get_euclidean_distances(arr, center, eyes, jays):
    h, w = arr.shape
    i, j = center
    i = min(max(i, 0), h - 1)
    j = min(max(j, 0), w - 1)
    return [(y - i)**2 + (x - j)**2 for y, x in zip(eyes, jays)]


def get_inds_in_sorted_order(arr, eyes, jays, dists, getChunks=True):
    order = sorted(range(len(dists)), key=lambda k: dists[k])
    potentials = [arr[eyes[i], jays[i]] for i in order]
    if getChunks:
        srted_dists = sorted(dists)
        curr = srted_dists[0]
        inds = [0]
        for i, d in enumerate(srted_dists):
            if d != curr:
                inds.append(i)
                curr = d
        inds.append(len(srted_dists))
        return potentials, inds
    return potentials


def get_nearest_cls(mask, point, r=5, debug=False):
    if debug:
        mask2 = mask + 0.0
        uniques = np.unique(mask2)
        for i, u in enumerate(uniques):
            mask2[mask2 == uniques[i]] = float(i)
        mask2 = (mask2 / mask2.max() * 255.0).astype(np.uint8)
        im = Image.fromarray(mask2)
        im = visualization.drawPoint(im, point[0], point[1])
        # from pylab import cm
        # cmap = cm.datad['Paired']
        # plt.imshow(im, cmap='tab20')
        # plt.show()
    eyes, jays = get_valid_circle_indices(mask, point, r)
    dists = get_euclidean_distances(mask, point, eyes, jays)
    potentials, inds = get_inds_in_sorted_order(mask, eyes, jays, dists)
    if debug:
        print(potentials)
    for i in range(len(inds) - 1):
        # check that 255 isn't an instance id
        vals = [v for v in potentials[inds[i]:inds[i + 1]]
                if v not in [0, 100.0, 255]]
        if len(vals) == 0:
            continue
        choice = max(set(vals), key=vals.count)
        return choice
    if r > 40 or debug:
        return 100.0  # just give up and return the silhouette
    return get_nearest_cls(mask, point, r * 2)


def silh2part(dImInfo, details, debug=False):
    if 'nearest_part_id' in dImInfo:
        return dImInfo['nearest_part_id']
    part_id = dImInfo['part_id']
    if part_id != '100':  # Only used for silhouettes
        return part_id
    obj_id = int(dImInfo['obj_id'])
    imid = dImInfo['imid']
    imid = int("".join(imid.split("_")))
    instance = int(dImInfo['inst_id'])
    parts = details.getMask(img=imid, cat=obj_id, instance=instance)
    h, w = parts.shape
    i, j = int(dImInfo['yCoord']), int(dImInfo['xCoord'])
    return get_nearest_cls(parts, (i, j), 5, debug)


def get_all_real_parts(dInfo, details, debug=False):
    for k, v in dInfo.items():
        part = silh2part(v, details, debug)
        dInfo[k]['nearest_part_id'] = int(part)


def make_dict_hr(d, ref, include_parts=True):
    hr = {}
    for k, v in d.items():
        key = ref[str(int(k))]['name'] if str(k) in ref else k
        if include_parts:
            vhr = {}
            for k1, v1 in v.items():
                key1 = ref[str(int(k))]['parts'][str(int(k1))] if str(
                    k1) in ref[str(k)]['parts']else k1
                vhr[key1] = v1
        else:
            vhr = v
        hr[key] = vhr
    return hr


def _make_dict_hr(D, ref):
    parts = set(list(ref.keys()))
    for k, v in D.items():
        if str(k) in parts:
            D[ref[str(k)]] = D.pop(k)
        if isinstance(v, dict):
            _make_dict_hr(v, ref)


def make_general_dict_hr(D, ref):
    objects = set(list(ref.keys()))
    for k, v in D.items():
        if str(k) in objects:
            D[ref[str(k)]['name']] = D.pop(k, None)
        if isinstance(v, dict):
            _make_dict_hr(v, ref[k]['parts'])


def check_occluded(imid, point, cls, inst=None, details=None):
    ''' Check whether a point (x,y) of class "cls" and instance id "inst" is in
        the background (e.g. if another Pascal object supersedes that region).
        The (hacky) logic relies on the accuracy of the semantic segmenation
        (which performs layering of overlapping instances). If the point is
        labelled as the class "cls" in the semantic image segmentation,
        we accept the object as being not occluded by another  Pascal
        object.
        "details" is a Details object.
    '''
    if isinstance(imid, str) or isinstance(imid, basestring):
        imid = int(imid[:4] + imid[5:])
    # Check whether point is occluded by a different class
    semantic_mask = details.getMask(img=imid)  # Get semantic segmentation mask
    x, y = int(point[1]), int(point[0])
    gt_class = int(semantic_mask[y, x])
    if gt_class != int(cls):
        #         print(int(semantic_mask[y,x]), int(cls))
        if gt_class == 0:
            #             plt.show()
            return False, (imid, point, cls)
        return True, gt_class  # Is occluded

    # TODO
    # Check whether point is occluded by a different instance
    # TODO
    # Check whether poin tis occluded by a different PART
    return False, 0
