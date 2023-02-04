
function b64toImageData(datas) {
    return new Promise(async function (resolve, reject) {
        // We create an image to receive the Data URI
        var img = document.createElement("img");

        // When the event "onload" is triggered we can resize the image.
        img.onload = function () {
            // We create a canvas and get its context.
            var canvas = document.createElement("canvas");
            var ctx = canvas.getContext("2d");

            // We set the dimensions at the wanted size.
            canvas.width = img.width;
            canvas.height = img.height;

            // We resize the image with the canvas method drawImage();
            ctx.drawImage(this, 0, 0, img.width, img.height);
            // get image data
            var imageDatas = ctx.getImageData(0, 0, img.width, img.height);

            // This is the return of the Promise
            resolve(imageDatas);
        };
        // add data to img
        img.src = datas;
    });
}

document.body.addEventListener("click", async (e) => {
    let el = e.target;
    if (el.id == "t-fg") {

        var fg = document.getElementById("t-fg").style.backgroundImage.slice(5, -2);

        const inverseMapper = {1: '0',
                                2: '2',
                                3: '4',
                                4: '8',
                                5: 'A',
                                6: 'D',
                                7: 'G',
                                8: 'H',
                                9: 'J',
                                10: 'K',
                                11: 'M',
                                12: 'N',
                                13: 'P',
                                14: 'R',
                                15: 'S',
                                16: 'T',
                                17: 'V',
                                18: 'W',
                                19: 'X',
                                20: 'Y'};

        // create a new session and load the the model.
        const session = await ort.InferenceSession.create('https://cors-anywhere.herokuapp.com/https://litter.catbox.moe/h72uqx.onnx', {
            executionProviders: ["cpu"],
        });

        var ImageFilters = {};
        ImageFilters.utils = {
            initSampleCanvas: function () {
                var _canvas = document.createElement("canvas"),
                    _context = _canvas.getContext("2d");

                _canvas.width = 0;
                _canvas.height = 0;

                this.getSampleCanvas = function () {
                    return _canvas;
                };
                this.getSampleContext = function () {
                    return _context;
                };
                this.createImageData = _context.createImageData
                    ? function (w, h) {
                        return _context.createImageData(w, h);
                    }
                    : function (w, h) {
                        return new ImageData(w, h);
                    };
            },
            getSampleCanvas: function () {
                this.initSampleCanvas();
                return this.getSampleCanvas();
            },
            getSampleContext: function () {
                this.initSampleCanvas();
                return this.getSampleContext();
            },
            createImageData: function (w, h) {
                this.initSampleCanvas();
                return this.createImageData(w, h);
            },
            clamp: function (value) {
                return value > 255 ? 255 : value < 0 ? 0 : value;
            },
            buildMap: function (f) {
                for (var m = [], k = 0, v; k < 256; k += 1) {
                    m[k] = (v = f(k)) > 255 ? 255 : v < 0 ? 0 : v | 0;
                }
                return m;
            },
            applyMap: function (src, dst, map) {
                for (var i = 0, l = src.length; i < l; i += 4) {
                    dst[i] = map[src[i]];
                    dst[i + 1] = map[src[i + 1]];
                    dst[i + 2] = map[src[i + 2]];
                    dst[i + 3] = src[i + 3];
                }
            },
            mapRGB: function (src, dst, func) {
                this.applyMap(src, dst, this.buildMap(func));
            },
            getPixelIndex: function (x, y, width, height, edge) {
                if (x < 0 || x >= width || y < 0 || y >= height) {
                    switch (edge) {
                        case 1: // clamp
                            x = x < 0 ? 0 : x >= width ? width - 1 : x;
                            y = y < 0 ? 0 : y >= height ? height - 1 : y;
                            break;
                        case 2: // wrap
                            x = (x %= width) < 0 ? x + width : x;
                            y = (y %= height) < 0 ? y + height : y;
                            break;
                        default: // transparent
                            return null;
                    }
                }
                return (y * width + x) << 2;
            },
            getPixel: function (src, x, y, width, height, edge) {
                if (x < 0 || x >= width || y < 0 || y >= height) {
                    switch (edge) {
                        case 1: // clamp
                            x = x < 0 ? 0 : x >= width ? width - 1 : x;
                            y = y < 0 ? 0 : y >= height ? height - 1 : y;
                            break;
                        case 2: // wrap
                            x = (x %= width) < 0 ? x + width : x;
                            y = (y %= height) < 0 ? y + height : y;
                            break;
                        default: // transparent
                            return 0;
                    }
                }

                var i = (y * width + x) << 2;

                // ARGB
                return (
                    (src[i + 3] << 24) | (src[i] << 16) | (src[i + 1] << 8) | src[i + 2]
                );
            },
            getPixelByIndex: function (src, i) {
                return (
                    (src[i + 3] << 24) | (src[i] << 16) | (src[i + 1] << 8) | src[i + 2]
                );
            },
            /**
             * one of the most important functions in this library.
             * I want to make this as fast as possible.
             */
            copyBilinear: function (src, x, y, width, height, dst, dstIndex, edge) {
                var fx = x < 0 ? (x - 1) | 0 : x | 0, // Math.floor(x)
                    fy = y < 0 ? (y - 1) | 0 : y | 0, // Math.floor(y)
                    wx = x - fx,
                    wy = y - fy,
                    i,
                    nw = 0,
                    ne = 0,
                    sw = 0,
                    se = 0,
                    cx,
                    cy,
                    r,
                    g,
                    b,
                    a;

                if (fx >= 0 && fx < width - 1 && fy >= 0 && fy < height - 1) {
                    // in bounds, no edge actions required
                    i = (fy * width + fx) << 2;

                    if (wx || wy) {
                        nw =
                            (src[i + 3] << 24) |
                            (src[i] << 16) |
                            (src[i + 1] << 8) |
                            src[i + 2];

                        i += 4;
                        ne =
                            (src[i + 3] << 24) |
                            (src[i] << 16) |
                            (src[i + 1] << 8) |
                            src[i + 2];

                        i = i - 8 + (width << 2);
                        sw =
                            (src[i + 3] << 24) |
                            (src[i] << 16) |
                            (src[i + 1] << 8) |
                            src[i + 2];

                        i += 4;
                        se =
                            (src[i + 3] << 24) |
                            (src[i] << 16) |
                            (src[i + 1] << 8) |
                            src[i + 2];
                    } else {
                        // no interpolation required
                        dst[dstIndex] = src[i];
                        dst[dstIndex + 1] = src[i + 1];
                        dst[dstIndex + 2] = src[i + 2];
                        dst[dstIndex + 3] = src[i + 3];
                        return;
                    }
                } else {
                    // edge actions required
                    nw = this.getPixel(src, fx, fy, width, height, edge);

                    if (wx || wy) {
                        ne = this.getPixel(src, fx + 1, fy, width, height, edge);
                        sw = this.getPixel(src, fx, fy + 1, width, height, edge);
                        se = this.getPixel(src, fx + 1, fy + 1, width, height, edge);
                    } else {
                        // no interpolation required
                        dst[dstIndex] = (nw >> 16) & 0xff;
                        dst[dstIndex + 1] = (nw >> 8) & 0xff;
                        dst[dstIndex + 2] = nw & 0xff;
                        dst[dstIndex + 3] = (nw >> 24) & 0xff;
                        return;
                    }
                }

                cx = 1 - wx;
                cy = 1 - wy;
                r =
                    (((nw >> 16) & 0xff) * cx + ((ne >> 16) & 0xff) * wx) * cy +
                    (((sw >> 16) & 0xff) * cx + ((se >> 16) & 0xff) * wx) * wy;
                g =
                    (((nw >> 8) & 0xff) * cx + ((ne >> 8) & 0xff) * wx) * cy +
                    (((sw >> 8) & 0xff) * cx + ((se >> 8) & 0xff) * wx) * wy;
                b =
                    ((nw & 0xff) * cx + (ne & 0xff) * wx) * cy +
                    ((sw & 0xff) * cx + (se & 0xff) * wx) * wy;
                a =
                    (((nw >> 24) & 0xff) * cx + ((ne >> 24) & 0xff) * wx) * cy +
                    (((sw >> 24) & 0xff) * cx + ((se >> 24) & 0xff) * wx) * wy;

                dst[dstIndex] = r > 255 ? 255 : r < 0 ? 0 : r | 0;
                dst[dstIndex + 1] = g > 255 ? 255 : g < 0 ? 0 : g | 0;
                dst[dstIndex + 2] = b > 255 ? 255 : b < 0 ? 0 : b | 0;
                dst[dstIndex + 3] = a > 255 ? 255 : a < 0 ? 0 : a | 0;
            },
            /**
             * @param r 0 <= n <= 255
             * @param g 0 <= n <= 255
             * @param b 0 <= n <= 255
             * @return Array(h, s, l)
             */
            rgbToHsl: function (r, g, b) {
                r /= 255;
                g /= 255;
                b /= 255;

                //        var max = Math.max(r, g, b),
                //            min = Math.min(r, g, b),
                var max = r > g ? (r > b ? r : b) : g > b ? g : b,
                    min = r < g ? (r < b ? r : b) : g < b ? g : b,
                    chroma = max - min,
                    h = 0,
                    s = 0,
                    // Lightness
                    l = (min + max) / 2;

                if (chroma !== 0) {
                    // Hue
                    if (r === max) {
                        h = (g - b) / chroma + (g < b ? 6 : 0);
                    } else if (g === max) {
                        h = (b - r) / chroma + 2;
                    } else {
                        h = (r - g) / chroma + 4;
                    }
                    h /= 6;

                    // Saturation
                    s = l > 0.5 ? chroma / (2 - max - min) : chroma / (max + min);
                }

                return [h, s, l];
            },
            /**
             * @param h 0.0 <= n <= 1.0
             * @param s 0.0 <= n <= 1.0
             * @param l 0.0 <= n <= 1.0
             * @return Array(r, g, b)
             */
            hslToRgb: function (h, s, l) {
                var m1,
                    m2,
                    hue,
                    r,
                    g,
                    b,
                    rgb = [];

                if (s === 0) {
                    r = g = b = (l * 255 + 0.5) | 0;
                    rgb = [r, g, b];
                } else {
                    if (l <= 0.5) {
                        m2 = l * (s + 1);
                    } else {
                        m2 = l + s - l * s;
                    }

                    m1 = l * 2 - m2;
                    hue = h + 1 / 3;

                    var tmp;
                    for (var i = 0; i < 3; i += 1) {
                        if (hue < 0) {
                            hue += 1;
                        } else if (hue > 1) {
                            hue -= 1;
                        }

                        if (6 * hue < 1) {
                            tmp = m1 + (m2 - m1) * hue * 6;
                        } else if (2 * hue < 1) {
                            tmp = m2;
                        } else if (3 * hue < 2) {
                            tmp = m1 + (m2 - m1) * (2 / 3 - hue) * 6;
                        } else {
                            tmp = m1;
                        }

                        rgb[i] = (tmp * 255 + 0.5) | 0;

                        hue -= 1 / 3;
                    }
                }

                return rgb;
            },
        };

        // Bilinear
        function ResizeBilinear(srcImageData, width, height) {
            var srcPixels = srcImageData.data,
                srcWidth = srcImageData.width,
                srcHeight = srcImageData.height,
                srcLength = srcPixels.length,
                dstImageData = ImageFilters.utils.createImageData(width, height),
                dstPixels = dstImageData.data;

            var xFactor = srcWidth / width,
                yFactor = srcHeight / height,
                dstIndex = 0,
                x,
                y;

            for (y = 0; y < height; y += 1) {
                for (x = 0; x < width; x += 1) {
                    ImageFilters.utils.copyBilinear(
                        srcPixels,
                        x * xFactor,
                        y * yFactor,
                        srcWidth,
                        srcHeight,
                        dstPixels,
                        dstIndex,
                        0
                    );
                    dstIndex += 4;
                }
            }

            return dstImageData;
        }

        // NN
        function ResizeNearestNeighbor(srcImageData, width, height) {
            var srcPixels = srcImageData.data,
                srcWidth = srcImageData.width,
                srcHeight = srcImageData.height,
                srcLength = srcPixels.length,
                dstImageData = ImageFilters.utils.createImageData(width, height),
                dstPixels = dstImageData.data;

            var xFactor = srcWidth / width,
                yFactor = srcHeight / height,
                dstIndex = 0,
                srcIndex,
                x,
                y,
                offset;

            for (y = 0; y < height; y += 1) {
                offset = ((y * yFactor) | 0) * srcWidth;

                for (x = 0; x < width; x += 1) {
                    srcIndex = (offset + x * xFactor) << 2;

                    dstPixels[dstIndex] = srcPixels[srcIndex];
                    dstPixels[dstIndex + 1] = srcPixels[srcIndex + 1];
                    dstPixels[dstIndex + 2] = srcPixels[srcIndex + 2];
                    dstPixels[dstIndex + 3] = srcPixels[srcIndex + 3];
                    dstIndex += 4;
                }
            }

            return dstImageData;
        }

        function ConvolutionFilter(
            srcImageData,
            matrixX,
            matrixY,
            matrix,
            divisor,
            bias,
            preserveAlpha,
            clamp,
            color,
            alpha
        ) {
            var srcPixels = srcImageData.data,
                srcWidth = srcImageData.width,
                srcHeight = srcImageData.height,
                srcLength = srcPixels.length,
                dstImageData = ImageFilters.utils.createImageData(srcWidth, srcHeight),
                dstPixels = dstImageData.data;

            divisor = divisor || 1;
            bias = bias || 0;

            // default true
            preserveAlpha !== false && (preserveAlpha = true);
            clamp !== false && (clamp = true);

            color = color || 0;
            alpha = alpha || 0;

            var index = 0,
                rows = matrixX >> 1,
                cols = matrixY >> 1,
                clampR = (color >> 16) & 0xff,
                clampG = (color >> 8) & 0xff,
                clampB = color & 0xff,
                clampA = alpha * 0xff;

            for (var y = 0; y < srcHeight; y += 1) {
                for (var x = 0; x < srcWidth; x += 1, index += 4) {
                    var r = 0,
                        g = 0,
                        b = 0,
                        a = 0,
                        replace = false,
                        mIndex = 0,
                        v;

                    for (var row = -rows; row <= rows; row += 1) {
                        var rowIndex = y + row,
                            offset;

                        if (0 <= rowIndex && rowIndex < srcHeight) {
                            offset = rowIndex * srcWidth;
                        } else if (clamp) {
                            offset = y * srcWidth;
                        } else {
                            replace = true;
                        }

                        for (var col = -cols; col <= cols; col += 1) {
                            var m = matrix[mIndex++];

                            if (m !== 0) {
                                var colIndex = x + col;

                                if (!(0 <= colIndex && colIndex < srcWidth)) {
                                    if (clamp) {
                                        colIndex = x;
                                    } else {
                                        replace = true;
                                    }
                                }

                                if (replace) {
                                    r += m * clampR;
                                    g += m * clampG;
                                    b += m * clampB;
                                    a += m * clampA;
                                } else {
                                    var p = (offset + colIndex) << 2;
                                    r += m * srcPixels[p];
                                    g += m * srcPixels[p + 1];
                                    b += m * srcPixels[p + 2];
                                    a += m * srcPixels[p + 3];
                                }
                            }
                        }
                    }

                    dstPixels[index] =
                        (v = r / divisor + bias) > 255 ? 255 : v < 0 ? 0 : v | 0;
                    dstPixels[index + 1] =
                        (v = g / divisor + bias) > 255 ? 255 : v < 0 ? 0 : v | 0;
                    dstPixels[index + 2] =
                        (v = b / divisor + bias) > 255 ? 255 : v < 0 ? 0 : v | 0;
                    dstPixels[index + 3] = preserveAlpha
                        ? srcPixels[index + 3]
                        : (v = a / divisor + bias) > 255
                            ? 255
                            : v < 0
                                ? 0
                                : v | 0;
                }
            }

            return dstImageData;
        }

        // * @ param strength 1 <= n <= 4
        function GaussianBlur(srcImageData, strength) {
            var size, matrix, divisor;

            switch (strength) {
                case 2:
                    size = 5;
                    matrix = [
                        1, 1, 2, 1, 1, 1, 2, 4, 2, 1, 2, 4, 8, 4, 2, 1, 2, 4, 2, 1, 1, 1, 2,
                        1, 1,
                    ];
                    divisor = 52;
                    break;
                case 3:
                    size = 7;
                    matrix = [
                        1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 4, 2, 2, 1, 2, 2, 4, 8, 4, 2, 2, 2, 4,
                        8, 16, 8, 4, 2, 2, 2, 4, 8, 4, 2, 2, 1, 2, 2, 4, 2, 2, 1, 1, 1, 2,
                        2, 2, 1, 1,
                    ];
                    divisor = 140;
                    break;
                case 4:
                    size = 15;
                    matrix = [
                        2, 2, 3, 4, 5, 5, 6, 6, 6, 5, 5, 4, 3, 2, 2, 2, 3, 4, 5, 7, 7, 8, 8,
                        8, 7, 7, 5, 4, 3, 2, 3, 4, 6, 7, 9, 10, 10, 11, 10, 10, 9, 7, 6, 4,
                        3, 4, 5, 7, 9, 10, 12, 13, 13, 13, 12, 10, 9, 7, 5, 4, 5, 7, 9, 11,
                        13, 14, 15, 16, 15, 14, 13, 11, 9, 7, 5, 5, 7, 10, 12, 14, 16, 17,
                        18, 17, 16, 14, 12, 10, 7, 5, 6, 8, 10, 13, 15, 17, 19, 19, 19, 17,
                        15, 13, 10, 8, 6, 6, 8, 11, 13, 16, 18, 19, 20, 19, 18, 16, 13, 11,
                        8, 6, 6, 8, 10, 13, 15, 17, 19, 19, 19, 17, 15, 13, 10, 8, 6, 5, 7,
                        10, 12, 14, 16, 17, 18, 17, 16, 14, 12, 10, 7, 5, 5, 7, 9, 11, 13,
                        14, 15, 16, 15, 14, 13, 11, 9, 7, 5, 4, 5, 7, 9, 10, 12, 13, 13, 13,
                        12, 10, 9, 7, 5, 4, 3, 4, 6, 7, 9, 10, 10, 11, 10, 10, 9, 7, 6, 4,
                        3, 2, 3, 4, 5, 7, 7, 8, 8, 8, 7, 7, 5, 4, 3, 2, 2, 2, 3, 4, 5, 5, 6,
                        6, 6, 5, 5, 4, 3, 2, 2,
                    ];
                    divisor = 2044;
                    break;
                default:
                    size = 3;
                    matrix = [1, 2, 1, 2, 4, 2, 1, 2, 1];
                    divisor = 16;
                    break;
            }
            return ConvolutionFilter(
                srcImageData,
                size,
                size,
                matrix,
                divisor,
                0,
                false
            );
        }

        // base image.
        var base_img = await b64toImageData(fg);
   
        // resize opts
        //var newImage = await resizedataURL(str, 300, 80);

        // nearest neighbors resize
        var nnImg = ResizeNearestNeighbor(base_img, 300, 80);

        //var blImg = ResizeBilinear(base_img, 300, 80);

        // gauss blur
        var gauss = GaussianBlur(nnImg, 3);

        // send to array
        const inputData2 = Float32Array.from(gauss.data);

        function imageDataToTensor(data, dims) {
            // 1a. Extract the R, G, and B channels from the data
            const [R, G, B] = [[], [], []];
            for (let i = 0; i < data.length; i += 4) {
                R.push(data[i]);
                G.push(data[i + 1]);
                B.push(data[i + 2]);
                // 2. skip data[i + 3] thus filtering out the alpha channel
            }
            // 1b. concatenate RGB ~= transpose [224, 224, 3] -> [3, 224, 224]
            const transposedData = R.concat(G).concat(B);

            // 3. convert to float32
            let i,
                l = transposedData.length; // length, we need this for the loop
            const float32Data = new Float32Array(1 * 1 * 80 * 300); // create the Float32Array for output
            for (i = 0; i < l; i++) {
                //float32Data[i] = transposedData[i] / 255.0;
                float32Data[i] = (transposedData[i] / 127.5) -1;
            }

            const inputTensor = {
                input: new ort.Tensor("float32", float32Data, dims),
            };
            return inputTensor;
        }

        // reshape
        var out1 = imageDataToTensor(inputData2, [1, 1, 80, 300]);

        // run model
        const results = await session.run(out1);

        // get predictions
        var output_arr = results.output.data;

        // filter repeat obs
        let res = output_arr.filter((x, i) => x !== output_arr[i - 1]);

        // filter blank value
        var removeArr = [0n];

        // filter blank execution
        var resultArr = res.filter((x) => !removeArr.includes(x));
        console.log('recoded out', resultArr);

        // turn to string
        var resultArr = resultArr.join().split(",");

        // map to char
        var replaced = resultArr.map((el) => inverseMapper[el]);

        // set string
        var replaced = replaced.toString();

        // remove commas
        var replaced = replaced.replaceAll(",", "");

        // write captcha to window
        console.log('replaced', replaced);

        // attach to field
        var resp = document.getElementById('t-resp');
        resp.value = replaced;

    }
});

