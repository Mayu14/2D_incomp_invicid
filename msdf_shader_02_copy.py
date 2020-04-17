# coding: utf-8
import moderngl
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from objloader import Obj
import cv2


def check():
    ctx = moderngl.create_standalone_context()
    buf = ctx.buffer(b'Hello World!')
    print(buf.read())

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def test_3d_rendering(fname, render_size=10000, fromObj=False, imgshow=False, RBG=True, returnCV2img=False):
    """

    :param fname:   picture's fname or numpy array (rgb img)
    :param fromObj:
    :param imgshow:
    :return:
    """

    ctx = moderngl.create_standalone_context()

    if fromObj:
        obj = Obj.open('sample.obj')
        vbo = ctx.buffer(obj.pack('vx vy vz tx ty'))
    else:
        vertices = np.array([
            0.5, 0.5, 0.0, 1.0, 0.0, 0.0,
            -0.5, 0.5, 0.0, 0.0, 1.0, 0.0,
            -0.5, -0.5, 0.0, 0.0, 0.0, 1.0
        ])
        # Vertex Buffer Object
        vbo = ctx.buffer(vertices.astype('float32').tobytes())

    prog = ctx.program(
        vertex_shader="""
            #version 330
            in vec3 in_vert;
            in vec2 in_text;
            out vec2 pos;
            uniform vec3 scale;
            # 具体的な値はcpu側で宣言しておき、それを引っ張ってくる

            void main() {
                pos = in_text;
                gl_Position = vec4(in_vert * scale, 1.0);
            }
        """,
        fragment_shader="""
            #version 330
            in vec2 pos; 
            out vec4 color;
            uniform sampler2D msdf;
            uniform float pxRange;
            uniform vec4 bgColor;
            uniform vec4 fgColor;

            float median(float r, float g, float b) {
                return max(min(r, g), min(max(r, g), b));
            }

            void main() {
                vec2 msdfUnit = pxRange/vec2(textureSize(msdf, 0));
                vec3 sample = texture(msdf, pos).rgb;
                float sigDist = median(sample.r, sample.g, sample.b) - 0.5;
                sigDist *= dot(msdfUnit, 0.5/fwidth(pos));
                float opacity = clamp(sigDist + 0.5, 0.0, 1.0);
                color = mix(bgColor, fgColor, opacity);
            }
        """
    )
    # prog内で宣言したuniformパラメータを属性値として指定
    prog["scale"].value = (2.0, 2.0, 1.0)
    prog["pxRange"].value = 1.0
    prog["bgColor"].value = (0.0, 0.0, 0.0, 1.0)
    prog["fgColor"].value = (1.0, 1.0, 1.0, 1.0)
    # Vertex Array Object
    # vao = ctx.simple_vertex_array(prog, vbo, 'in_vert', "in_color")
    vao = ctx.simple_vertex_array(prog, vbo, 'in_vert', "in_text")

    if str(type(fname)) == "<class 'PIL.Image.Image'>":
        textureimg = fname  # numpy array
    else:
        textureimg = Image.open(fname).transpose(Image.FLIP_TOP_BOTTOM).convert('RGB')

    texture = ctx.texture(textureimg.size, 3, textureimg.tobytes())
    texture.build_mipmaps()
    texture.use()

    # Frame Buffer Object
    fbo = ctx.simple_framebuffer((render_size, render_size))
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 1.0)

    vao.render()

    myimg = Image.frombytes('RGB', fbo.size, fbo.read(), "raw", "RGB", 0, -1)
    if imgshow:
        plt.imshow(myimg)
        plt.show()
    if not returnCV2img:
        if RBG:
            return np.asarray(myimg)[:, :, ::-1]  # RGB(Pillow) -> RBG(OpenCV2)
        else:
            return np.asarray(myimg)[:, :, :]  # RGB(Pillow)
    else:
        return pil2cv(myimg)


def extract_edge(my_img):
    def draw_contours(ax, img, contours):
        ax.imshow(img)  # 画像を表示する。
        ax.set_axis_off()

        for i, cnt in enumerate(contours):
            # 形状を変更する。(NumPoints, 1, 2) -> (NumPoints, 2)
            cnt = cnt.squeeze(axis=1)
            # 輪郭の点同士を結ぶ線を描画する。
            ax.add_patch(Polygon(cnt, color="b", fill=None, lw=2))
            # 輪郭の点を描画する。
            ax.plot(cnt[:, 0], cnt[:, 1], "ro", mew=0, ms=4)
            # 輪郭の番号を描画する。
            ax.text(cnt[0][0], cnt[0][1], i, color="orange", size="20")

    # エッジ抽出
    imgray = cv2.cvtColor(my_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    fig, ax = plt.subplots(figsize=(8, 8))
    draw_contours(ax, my_img, contours)
    plt.show()

def openPILimage(fname, flip=True):
    if flip:
        return Image.open(fname).transpose(Image.FLIP_TOP_BOTTOM).convert('RGB')
    else:
        return Image.open(fname).convert('RGB')

def openCV2image(fname):
    return cv2.imread(fname)

def test_accuracy():
    fname_head = "G:\\Toyota\\Data\\grid_vtk\\NACA4\\"
    fname_foot = "_i_4_NACA2611_aoa004.png"
    test = []
    fname_mid = []
    for i in range(4, 12):
        test.append(2**i)
        if i < 8:
            fname_mid.append(str(test[i-4]) + "\\" + str(test[i-4]) + "_i")
        else:
            fname_mid.append(("test") + "\\" + str(test[i - 4]) + "_c")
    print(fname_head + fname_mid[-1] + fname_foot)
    img_b = openPILimage(fname_head + fname_mid[-1] + fname_foot)
    img_b = test_3d_rendering(img_b, fromObj=True, imgshow=False, returnCV2img=True)

    for i in range(4, 11):
        img_t = openPILimage(fname_head + fname_mid[i-4] + fname_foot)
        img_t = test_3d_rendering(img_t, fromObj=True, imgshow=False, returnCV2img=True)
        img_diff = cv2.absdiff(img_b, img_t)
        error = (np.average(img_diff))
        # fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        # fgmask = fgbg.apply(img_b)
        # fgmask = fgbg.apply(img_t)
        # 表示
        cv2.imwrite("msdf_px" + str(test[i-4]) + "_" + str(error) + ".png",img_diff)

    exit()

if __name__ == '__main__':
    # check()
    fname_t = "G:\\Toyota\\Data\\grid_vtk\\NACA4\\128\\128_i_i_4_NACA2611_aoa004.png"
    fname_o = "G:\\Toyota\\Data\\grid_vtk\\NACA4\\128\\128_i_i_4_NACA2611_aoa004.bmp"
    test_accuracy()
    img = openPILimage(fname_t)
    myimg = test_3d_rendering(img, fromObj=True, imgshow=True)
    extract_edge(myimg)