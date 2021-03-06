import math

from sklearn.cluster import KMeans, DBSCAN
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import cv2
import scipy
import scipy.misc
import scipy.cluster

classified_colours = {
    0: (0, 0, 0),
    1: (255, 255, 255),
    2: (125, 200, 255)
}


def gen_xywh_from_box(box):
    x = int(box[1])  
    y = int(box[0])  
    w = int(box[3]-box[1])
    h = int(box[2]-box[0])
    if x < 0 :
        w = w + x
        x = 0
    if y < 0 :
        h = h + y
        y = 0 
    return x, y, w, h 


def _int_(tup):
    return [int(tup_val) for tup_val in tup]


def remove_background_and_average_colour(image_np, NUM_CLUSTERS=5):
    #only_coloured_pixels = np.array([pixel for pixel in image_np.reshape((-1, 3)) if pixel.tolist() != [0, 0, 0]])
    #return np.average(only_coloured_pixels, axis=0).astype(int)

    shape = image_np.shape
    ar = image_np.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
    ar = np.array([pixel for pixel in ar if pixel.tolist() != [0, 0, 0]])
    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences
    index_max = scipy.argmax(counts)                    # find most frequent
    peak = tuple(_int_(codes[index_max]))
    return peak


def sort_by_lowest_translator(cluster_centers, n_clusters):
    """
    Sorts by lowest colour val to highest, to keep colour consistent on the same team
    """
    d = {}
    mean = np.mean(cluster_centers, axis=1)
    d[0] = np.argmin(mean)
    d[n_clusters - 1] = np.argmax(mean)
    if n_clusters == 3:
        d[1] = 3 - d[0] - d[2]
    return d


def classify_masks(masks, by="average_colour", n_clusters=2):
    colours = [mask.__dict__[by] for mask in masks]
    n_clusters = n_clusters if len(masks)>=3 else len(masks)

    if n_clusters == 0:
        return masks

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(colours)
    except:
        import ipdb
        ipdb.set_trace()

    translator = sort_by_lowest_translator(kmeans.cluster_centers_, n_clusters)
    for i, mask in enumerate(masks):
        mask.classify = translator[kmeans.labels_[i]]
        mask.drawn_colour = classified_colours[mask.classify]
        mask.kmeans_label = kmeans.labels_[i]

    return masks


def draw_classified_ellipses_around_masks(image, masks, by="drawn_colour"):
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw = ImageDraw.Draw(image_pil)
    for mask in masks:
        (y0, x0, y1, x1) = mask.rois
        player_height = y1 - y0
        y0 = y1 - player_height*0.2
        draw.ellipse([x0, y0, x1, y1], fill=mask.__dict__[by])

    np.copyto(image, np.array(image_pil))


def draw_line_from_distances(image, players_group, distances, colour=(255, 255, 255)):
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')

    draw = ImageDraw.Draw(image_pil)
    for distance in distances:
        n1 = int(distance[0])
        n2 = int(distance[1])
        draw.line(
                [ players_group[n1].center, players_group[n2].center ],
                fill=colour,
                width = 3
                )
    np.copyto(image, np.array(image_pil))
    return image


def draw_lines_between_classified_players(image, players, by="drawn_colour"):
    no_classes = len(classified_colours)
    for i in range(no_classes):
        players_group = players.filter_classify(i)
        n = len(players_group)
        distances = players_group.distances() 
        if len(distances) == 0:
            continue
        #distances = distances[:n-1]
        image = draw_line_from_distances(image, players_group, distances, colour=players_group[0].__dict__[by])


def apply_masks_to_image_np(image_np, masks):
    if len(masks) == 0:
        return image_np

    for i, mask in enumerate(masks):
        mask.average_colour = remove_background_and_average_colour(mask.masked_image_np)

    masks = classify_masks(masks, by="average_colour")
    draw_classified_ellipses_around_masks(image_np, masks)
    draw_lines_between_classified_players(image_np, masks)
    return image_np, masks

def _ceil_(num, to=10):
    return int(math.ceil(num/to)*to)


def draw_player_with_tracks(image_np, tracks, force=False, alpha=0.5):
    overlay = image_np.copy()

    for track in tracks:
        if (not track.is_confirmed() or track.time_since_update > 1) and not force:
            continue
        bbox = track.to_tlbr()
        center = (int((bbox[2] + bbox[0])/2), int(bbox[3]))
        x_length = _ceil_(bbox[2] - bbox[0])
        axes = (int(x_length/2), int(x_length/8)) # x length, y length
        thickness = int(x_length/8) #int(axes[1]/2)

        #cv2.rectangle(image_np, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), classified_colours[track.team_id], 2)
        cv2.ellipse(overlay, center, axes, 0, 0, 360, color=classified_colours[track.team_id], thickness=thickness)
        cv2.putText(overlay, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, classified_colours[track.team_id], 2)

    cv2.addWeighted(overlay, alpha, image_np, 1 - alpha, 0, image_np)


def load_image_into_numpy_array(image):
    try:
        image_np = np.array(image)
        (im_height, im_width, channels) = image_np.shape
    except:
        return image

    if channels > 3:
        image_np = image_np[:, :, :3]
    return image_np.astype(np.uint8)


def _pixel_is_black_(pixel):
    return all(pixel == [0, 0, 0])

def _pixel_is_black_wrapper(pixel):
    return pixel if not _pixel_is_black_(pixel) else None

def _mean_(l):
    return sum(l)/len(l)

def _stringify_colour(colour):
    return "{}, {}, {}".format(colour[0], colour[1], colour[2])

def classify_masks_with_hash(masks, n_clusters=2):
    if len(masks) == 0:
        return masks

    all_colours = []
    for i, mask in enumerate(masks):
        mask.flattened_colour = mask.upper_half_np.reshape((-1, 3))
        mask.flattened_colour = [pixel for pixel in mask.flattened_colour if not _pixel_is_black_(pixel)]
        all_colours.append(mask.flattened_colour)
    
    all_colours = np.concatenate(all_colours)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_colours)

    # lower number of colour = 0, so if first number is greater, flip labels
    if _mean_(kmeans.cluster_centers_[0]) > _mean_(kmeans.cluster_centers_[1]):
        kmeans.labels_ = [ 1 - val for val in kmeans.labels_]

    string_colours = [_stringify_colour(colour) for colour in all_colours]
    zip_colours = zip(string_colours, kmeans.labels_)
    colour_label_hashmap = dict(zip_colours)

    for mask in masks:
        mask.kmeans_label = round(_mean_([colour_label_hashmap[_stringify_colour(colour)] for colour in mask.flattened_colour]))
    return masks    

