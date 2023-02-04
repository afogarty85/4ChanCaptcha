import cv2
import base64
import pytesseract
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\afogarty\AppData\Local\Tesseract-OCR\tesseract.exe"
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
    '''
    params:
    background: array  - image as np array
    foreground: array  - image as np array
    x_offset: int - where to slide the background on the x axis
    y_offset: int - where to slide the background on the y axis
    '''

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
    
    params:
    img: array  - image as np array
    params: dict  - dictionary of parameterized options
    '''

    # gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('uint8')

    # blur
    blurred = cv2.medianBlur(gray, params['blur'])

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
    for i, offset in enumerate(range(0, max_offset, 1)):
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
        # filter to just len 1 characters
        conf_out = conf_out.loc[conf_out['len'] == 1].reset_index(drop=True)
        # and just good results
        conf_out = conf_out.loc[conf_out['conf'] >= 70].reset_index(drop=True)

        if len(conf_out) >= 1:
            # concat
            storage_df = pd.concat([storage_df, conf_out], axis=0)

    # return max result
    if len(storage_df) >= 1:
        storage_df = storage_df.loc[storage_df['conf'] == storage_df['conf'].max()].reset_index(drop=True)
        return storage_df


def extract_character(background, foreground, x, y, h, w, offset, resize, text, conf):
    '''
    Extract individual characters given pytesseract's findings
    '''
    # generate raw image
    background, foreground = prepare_b64_image(background, foreground)

    # set it to the high performing offset
    extracting_image = add_transparent_image(background=background,
                                foreground=foreground,
                                x_offset=offset,
                                y_offset=0)

    # set it to the expected resize
    extracting_image_scaled = rescale_image(extracting_image, resize)

    # crop image
    crop = extracting_image_scaled[y:y+h, x:x+w]

    # edit conf
    conf = str(conf).replace('.', '')[0:8]

    # gray
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # write to disk
    cv2.imwrite(fr"C:\Users\afogarty\Desktop\extracted_chars\{text}_{conf}.png", gray)


def parallel_search_fn(bg_set, fg_set, params, max_workers):
    # for each image in the dataset
    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for bg, fg in zip(bg_set, fg_set):
            future = executor.submit(ocr_sliding_image, max_offset=60, bg=bg, fg=fg, params=params)
            futures.append(future)

    # unpack results
    print(f'Starting to unpack this many results: {len(futures)}')
    for future in futures:
        result = future.result()
        if result is not None:
            # extract chars
            for i in range(len(result)):
                extract_character(background=result.iloc[i]['bg_name'],
                                  foreground=result.iloc[i]['fg_name'],
                                  x=result.iloc[i]['left'],
                                  y=result.iloc[i]['top'],
                                  h=result.iloc[i]['height'],
                                  w=result.iloc[i]['width'],
                                  offset=result.iloc[i]['offset'],
                                  resize=params['resize'],
                                  text=result.iloc[i]['text'],
                                  conf=result.iloc[i]['conf']
                                  )



# # shapes
# shape1_ = cv2.MORPH_RECT
# shape2_ = cv2.MORPH_CROSS
# shape3_ = cv2.MORPH_ELLIPSE

if __name__ == '__main__':
    # find max workers
    _ = ProcessPoolExecutor()
    max_workers = _._max_workers
    del _
#
    # sample set
    captcha_set = pd.read_csv(r"C:\Users\afogarty\Desktop\captcha\dataset\raw_base64_dataset.csv")

    bg_set = captcha_set['bg'].values
    fg_set = captcha_set['fg'].values

#     # set params
# # finished with loss -81321596.3750 and params dict_items([('at_window', 33), ('b_l', 23), ('b_r', 5), ('c', 33), ('config', "--psm 6 --oem 1 -c tessedit_char_whitelist=' ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'"), ('contour_area', 420), ('e_l', 2), ('e_r', 4), ('morph_shape', 2), ('resize', 340), ('s_x', 3), ('s_y', 4), ('workers', 20)])
    params = {'at_window': 35,
              'b_l': 15,
              'b_r': 5,
              'c': 29,
              'blur': 5,
              'config': '--psm 6 --oem 1 -c tessedit_char_whitelist=" ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"',
              'contour_area': 450,
              'e_l': 4,
              'e_r': 4,
              'morph_shape': cv2.MORPH_CROSS,
              'resize': 170,
              's_x': 2,
              's_y': 1,
              'max_offset': 50}

    # run ops
    print(f'Starting operations using this many workers: {max_workers}')
    parallel_search_fn(bg_set, fg_set, params, max_workers)

    # report conclusion
    print('Operations finished!')









#
