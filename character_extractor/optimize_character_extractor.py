import cv2
import base64
import pytesseract
import numpy as np
import pandas as pd
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.early_stop import no_progress_loss
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\afogarty\AppData\Local\Tesseract-OCR\tesseract.exe"
#pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"
# white text on black background


def prepare_b64_image(background, foreground):
    '''
    params:
    background: string  - base64 image as text
    foreground: string  - base64 image as text
    '''
    # bg
    background = base64.b64decode(str(background))
    background = np.frombuffer(background, np.uint8)
    background = cv2.imdecode(background, cv2.IMREAD_UNCHANGED)

    # fg
    foreground = base64.b64decode(str(foreground))
    foreground = np.frombuffer(foreground, np.uint8)
    foreground = cv2.imdecode(foreground, cv2.IMREAD_UNCHANGED)
    return background, foreground


def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)

    # gen copies
    fg, bg = foreground.copy(), background.copy()

    fg = fg[fg_y:fg_y + h, fg_x:fg_x + w]
    bg_subsection = bg[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = fg[:, :, :3]
    alpha_channel = fg[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = bg_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    bg[bg_y:bg_y + h, bg_x:bg_x + w] = composite

    # set to float32
    composite = np.float32(composite)
    return composite


def captcha_preprocessing(img, params):
    '''
    parameterized processing

    '''

    # gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('uint8')

    # blur
    blurred = cv2.medianBlur(gray, 5)

    # thresh
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, params['at_window'], params['c'])

    # Find contours and remove small noise; this is an option to just get a single char or so
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < params['contour_area']:
            cv2.drawContours(thresh, [c], -1, 0, -1)

    # cv.MORPH_RECT, cv.MORPH_CROSS, cv.MORPH_ELLIPSE
    kernel = cv2.getStructuringElement(params['morph_shape'], (params['e_l'], params['e_r']))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # another blur; not bad (5, 5), 3, 3)
    blurred2 = cv2.GaussianBlur(opening, (params['b_l'], params['b_r']), params['s_x'], params['s_y'])

    # crop edge noise
    crop = remove_noise(im_array=blurred2, lower=5, upper=95)

    # resize
    resized = rescale_image(crop, params['resize'])
    return resized


def remove_noise(im_array, lower, upper):
    '''
    Remove noise from image by cropping areas with few black pixels
    '''
    # loc of black pixels
    indices = np.where(im_array == [0])[0]  # y-axis

    # get histogram over indices
    y, x = np.histogram(indices, density=True)

    # get percentiles
    lower_p = np.percentile(indices, lower).astype(int)
    upper_p = np.percentile(indices, upper).astype(int)

    # crop image
    crop = im_array[lower_p:upper_p:, :]
    return crop


def rescale_image(im_array, percent):
    '''
    rescale image
    '''
    # percent of original size
    width = int(im_array.shape[1] * percent / 100)
    height = int(im_array.shape[0] * percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(im_array, dim, interpolation=cv2.INTER_NEAREST)
    return resized


def ocr_sliding_image(max_offset, bg, fg, params):
    # storage
    storage_df = pd.DataFrame()
    # prepare b64 image
    background, foreground = prepare_b64_image(background=bg,
                                               foreground=fg)

    # loop
    for i, offset in enumerate(range(0, max_offset, 2)):
        '''
        offset background with foreground and run pytesseract to determine
        individual characters and positions for cropping
        '''
        # run image joining at offset
        out = add_transparent_image(background=background,
                                    foreground=foreground,
                                    x_offset=0+offset,
                                    y_offset=0)

        try:
            # run pre-processing
            captcha_out = captcha_preprocessing(img=out, params=params)

        except Exception as e:
            print('exception triggered', e)
            print(f'params for exception: {params}')
            continue
        # get conf
        conf_out = pytesseract.image_to_data(captcha_out,
                                             lang='eng',
                                             config=params['config'],
                                             output_type='data.frame')

        # filter nan
        conf_out = conf_out.loc[conf_out['text'].notna()].copy()
        # go int if float
        # but handle np.inf situation
        conf_out.replace(np.inf, 0, inplace=True)
        conf_out['text'] = conf_out['text'].map(lambda x: int(x) if isinstance(x, float) else x)
        # set result to strings in case pandas converts
        conf_out['text'] = conf_out['text'].astype(str)
        # remoev periods
        conf_out['text'] = conf_out['text'].str.replace('.', '', regex=False)
        # strip white space
        conf_out['text'] = conf_out['text'].str.strip()
        # get lens
        conf_out['len'] = conf_out['text'].map(lambda x: len(x))
        # add in offset
        conf_out['offset'] = offset
        # add in filename
        conf_out['bg_name'] = bg
        conf_out['fg_name'] = fg
        # extract results
        conf_out = conf_out[['left', 'top', 'width', 'height', 'conf', 'text', 'offset', 'bg_name', 'fg_name', 'len']]
        if len(conf_out) >= 1:
            # concat
            storage_df = pd.concat([storage_df, conf_out], axis=0)
        else:
            continue

    # return max result
    if len(storage_df) >= 1:
        return storage_df
    else:
        # otherwise, still return a DF, but with zero
        return pd.DataFrame({'conf': [0], 'offset': [0], 'len': [0], 'width': [0], 'height': [0]})


def train_fn(params):
    '''
    HyperOpt with parallelized processing
    '''
    for key, value in params.items():
        try:
            params[key] = int(value)
        except Exception as e:
            params[key] = value

    # iterate over selection
    loss = 0.0
    # for each image in the dataset #  ThreadPoolExecutor; ProcessPoolExecutor
    futures = []
    with ProcessPoolExecutor(max_workers=params['workers']) as executor:
        for bg, fg in zip(bg_set, fg_set):
            future = executor.submit(ocr_sliding_image, max_offset=60, bg=bg, fg=fg, params=params)
            futures.append(future)

    # unpack results
    for future in futures:
        result = future.result()
        if result is not None:
            # filter to single chars
            result = result.loc[result['len'] == 1].copy()
            # scale larger image results
            result['scaled'] = ((result['width'] + result['height']) / 2) * result['conf']
            # store the confidence
            loss += np.sum(result['scaled'])
    

    print(f'finished with loss {-loss:.4f} and params {params.items()}')

    return {'loss': -loss, 'status': STATUS_OK}


# shapes
shape1_ = cv2.MORPH_RECT
shape2_ = cv2.MORPH_CROSS
shape3_ = cv2.MORPH_ELLIPSE

# odd vals
odd_choices = [x for x in range(3, 100) if x % 2 != 0]
odd_guassian = [x for x in range(0, 25) if x % 2 != 0]

# odd_blur[1]
# # {'at_window': 61, 'b_l': 13, 'b_r': 7, 'blur': 5, 'c': -30.0, 'config': 0, 'contour_area': 420.0, 'e_l': 3.0, 'e_r': 3.0, 'morph_shape': cv2.MORPH_RECT, 'resize': 330.0, 's_x': 2.0, 's_y': 2.0, 'workers': 0}


# hyperopt;
search_space = {
                'workers': hp.choice('workers', [20]),
                'config': hp.choice('config', ["--psm 6 --oem 1 -c tessedit_char_whitelist=' ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'"]),
                'at_window': hp.choice('at_window', odd_choices),
                'c': hp.quniform('c', -40, 60, 1.5),
                'contour_area': hp.quniform('contour_area', 250, 550, 10),
                'morph_shape': hp.choice('morph_shape', [shape1_, shape2_, shape3_]),
                'e_l': hp.quniform('e_l', 1, 5, 1),
                'e_r': hp.quniform('e_r', 1, 5, 1),
                #'k2_l': hp.quniform('k2_l', 3, 10, 1),
                #'k2_r': hp.quniform('k2_r', 1, 4, 1),
                'b_l': hp.choice('b_l', odd_guassian),
                'b_r': hp.choice('b_r', odd_guassian),
                's_x': hp.quniform('s_x', 1, 5, 1),
                's_y': hp.quniform('s_y', 1, 5, 1),
                'resize': hp.quniform('resize', 40, 350, 10)
                }




if __name__ == '__main__':

    # sample set

    captcha_set = pd.read_csv(r"C:\Users\afogarty\Desktop\captcha\dataset\raw_base64_dataset.csv")
    captcha_set = captcha_set.sample(frac=0.25, replace=False, random_state=44).reset_index(drop=True)

    # turn to np array
    bg_set = captcha_set['bg'].values
    fg_set = captcha_set['fg'].values


    # run
    argmin = fmin(fn=train_fn,
                  space=search_space,
                  algo=tpe.suggest,
                  max_evals=700,
                  early_stop_fn=no_progress_loss(iteration_stop_count=40,
                                                 percent_increase=1.0)
                  )

    print(argmin)

