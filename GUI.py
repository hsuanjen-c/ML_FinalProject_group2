import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from torchvision.models import segmentation
from torch import nn

STYLE_MODELS = {
    "comic": "generator_full.pth",
    "beauty": "beaty_filter.pth",
    "3d": ".pth"
}

#陳宣任
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()
        self.encoder = nn.Sequential(
            self.block(in_channels, features, normalize=False),
            self.block(features, features * 2),
            self.block(features * 2, features * 4),
            self.block(features * 4, features * 8),
        )
        self.decoder = nn.Sequential(
            self.upblock(features * 8, features * 4),
            self.upblock(features * 4, features * 2),
            self.upblock(features * 2, features),
            nn.ConvTranspose2d(features, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def block(self, in_c, out_c, normalize=True):
        layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def upblock(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
generator_comic = torch.load("generator_full.pth", weights_only=False)

#李杰軒
class GeneratorUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

generator_beauty = torch.load('beaty_filter.pth',weights_only=False, map_location='cpu')

#李柏蓁
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

       
        self.res_blocks = nn.Sequential(
            *[self._res_block(256) for _ in range(2)]
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 3, kernel_size=7, padding=3),
            nn.Tanh()  
        )

    def _res_block(self, dim):
        return nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        res = self.res_blocks(x) + x  
        out = self.decoder(res)
        return out





class StyleTransferApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Style Transfer GUI")

        # 選圖片按鈕
        self.btn_load = tk.Button(root, text="載入圖片", command=self.load_image)
        self.btn_load.pack()


        # 下拉式風格選單
        self.style_var = tk.StringVar(root)
        self.style_var.set("選擇風格")
        self.style_menu = tk.OptionMenu(root, self.style_var, *STYLE_MODELS.keys())
        self.style_menu.pack()

        # 轉換按鈕
        self.btn_transfer = tk.Button(root, text="進行風格轉換", command=self.transfer_style)
        self.btn_transfer.pack()

        # 兩張圖片的Canvas
        self.canvas_orig = tk.Label(root)
        self.canvas_orig.pack(side="left")
        self.canvas_trans = tk.Label(root)
        self.canvas_trans.pack(side="right")
        

        self.image = None
        self.canvas_text = tk.Canvas(root, width=256, height=300)
        self.canvas_text.pack()

    def load_image(self):
        filepath = filedialog.askopenfilename()
        if filepath:
            self.image = Image.open(filepath).convert('RGB')
            self.display_image(self.image, self.canvas_orig)

    def display_image(self, image, canvas):
        im = image.resize((256, 256))
        photo = ImageTk.PhotoImage(im)
        canvas.config(image=photo)
        canvas.image = photo
        

    def transfer_style(self):
        if self.image is None:
            return
        # 將圖片轉換為Tensor
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        image_tensor = transform(self.image).unsqueeze(0)
        # 載入模型
        style = self.style_var.get()
        device = torch.device("cpu")
        if style == "comic":
            #陳宣任
            generator_comic.to(device)
            generator_comic.eval()
            with torch.no_grad():
                fake_style = generator_comic(image_tensor)
                img_pil = self.tensor_to_pil(fake_style)
            self.display_image(img_pil, self.canvas_trans)
            self.canvas_text.delete("all")
            self.canvas_text.create_text(128, 100, text="模型製作:陳宣任", fill="black", font=("Arial", 14))
        elif style == "beauty":
            #李杰軒
            generator_beauty.to(device)
            generator_beauty.eval()
            with torch.no_grad():
                fake_style = generator_beauty(image_tensor)
                img_pil = self.tensor_to_pil(fake_style)
            self.display_image(img_pil, self.canvas_trans)
            self.canvas_text.delete("all")
            self.canvas_text.create_text(128, 100, text="模型製作:李杰軒", fill="black", font=("Arial", 14))
        elif style == "3d":
            #李柏蓁
            generator_3d = Generator()  # 初始化
            generator_3d.load_state_dict(torch.load('generator_a_weights.pth', map_location='cpu'))
            generator_3d.eval()
            with torch.no_grad():
                fake_style = generator_3d(image_tensor)
                img_pil = self.tensor_to_pil(fake_style)
            self.display_image(img_pil, self.canvas_trans)
            self.canvas_text.delete("all")
            self.canvas_text.create_text(128, 100, text="模型製作:李柏蓁", fill="black", font=("Arial", 14))

    def tensor_to_pil(self, tensor):
        from torchvision.transforms import ToPILImage
        output = tensor.cpu().squeeze(0)
        output = (output * 0.5 + 0.5).clamp(0, 1)
        img_pil = ToPILImage()(output)
        img_pil = img_pil.resize((256, 256))
        return img_pil

# 執行 GUI
root = tk.Tk()
app = StyleTransferApp(root)
root.mainloop()
