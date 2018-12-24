from sklearn.cluster import KMeans
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw


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


def remove_background_and_average_colour(image_np):
    only_coloured_pixels = np.array([pixel for pixel in image_np.reshape((-1, 3)) if pixel.tolist() != [0, 0, 0]])
    return np.average(only_coloured_pixels, axis=0).astype(int)


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


def classify_masks(masks, by="average_colour"):
    colours = [mask.__dict__[by] for mask in masks]
    n_clusters = 2 if len(masks)>=3 else len(masks)

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
    return image_np


