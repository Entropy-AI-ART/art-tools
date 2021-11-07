from torch import nn


class Encoder4(nn.Module):
    def __init__(self):
        super().__init__()
        # vgg
        # 224 x 224
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.reflec_pad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        # 226 x 226

        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.relu2 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflec_pad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu3 = nn.ReLU(inplace=True)
        # 224 x 224

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 112 x 112

        self.reflec_pad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu4 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflec_pad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu5 = nn.ReLU(inplace=True)
        # 112 x 112

        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 56 x 56

        self.reflec_pad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 0)
        self.relu6 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflec_pad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu7 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflec_pad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu8 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflec_pad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu9 = nn.ReLU(inplace=True)
        # 56 x 56

        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 28 x 28

        self.reflec_pad10 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = nn.Conv2d(256, 512, 3, 1, 0)
        self.relu10 = nn.ReLU(inplace=True)
        # 28 x 28

    def forward(self, x, sF=None, matrix11=None, matrix21=None, matrix31=None):
        output = {}
        out = self.conv1(x)
        out = self.reflec_pad1(out)
        out = self.conv2(out)
        output['r11'] = self.relu2(out)
        out = self.reflec_pad7(output['r11'])

        out = self.conv3(out)
        output['r12'] = self.relu3(out)

        output['p1'] = self.maxPool(output['r12'])
        out = self.reflec_pad4(output['p1'])
        out = self.conv4(out)
        output['r21'] = self.relu4(out)
        out = self.reflec_pad7(output['r21'])

        out = self.conv5(out)
        output['r22'] = self.relu5(out)

        output['p2'] = self.maxPool2(output['r22'])
        out = self.reflec_pad6(output['p2'])
        out = self.conv6(out)
        output['r31'] = self.relu6(out)
        if matrix31 is not None:
            feature3, transmatrix3 = matrix31(output['r31'], sF['r31'])
            out = self.reflec_pad7(feature3)
        else:
            out = self.reflec_pad7(output['r31'])
        out = self.conv7(out)
        output['r32'] = self.relu7(out)

        out = self.reflec_pad8(output['r32'])
        out = self.conv8(out)
        output['r33'] = self.relu8(out)

        out = self.reflec_pad9(output['r33'])
        out = self.conv9(out)
        output['r34'] = self.relu9(out)

        output['p3'] = self.maxPool3(output['r34'])
        out = self.reflec_pad10(output['p3'])
        out = self.conv10(out)
        output['r41'] = self.relu10(out)

        return output


class Decoder4(nn.Module):
    def __init__(self):
        super().__init__()
        # decoder
        self.reflec_pad11 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(512, 256, 3, 1, 0)
        self.relu11 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56

        self.reflec_pad12 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv12 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu12 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflec_pad13 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv13 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu13 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflec_pad14 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv14 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu14 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflec_pad15 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv15 = nn.Conv2d(256, 128, 3, 1, 0)
        self.relu15 = nn.ReLU(inplace=True)
        # 56 x 56

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 x 112

        self.reflec_pad16 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv16 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu16 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflec_pad17 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv17 = nn.Conv2d(128, 64, 3, 1, 0)
        self.relu17 = nn.ReLU(inplace=True)
        # 112 x 112

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        self.reflec_pad18 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv18 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu18 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflec_pad19 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv19 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, x):
        # decoder
        out = self.reflec_pad11(x)
        out = self.conv11(out)
        out = self.relu11(out)
        out = self.unpool(out)
        out = self.reflec_pad12(out)
        out = self.conv12(out)

        out = self.relu12(out)
        out = self.reflec_pad13(out)
        out = self.conv13(out)
        out = self.relu13(out)
        out = self.reflec_pad14(out)
        out = self.conv14(out)
        out = self.relu14(out)
        out = self.reflec_pad15(out)
        out = self.conv15(out)
        out = self.relu15(out)
        out = self.unpool2(out)
        out = self.reflec_pad16(out)
        out = self.conv16(out)
        out = self.relu16(out)
        out = self.reflec_pad17(out)
        out = self.conv17(out)
        out = self.relu17(out)
        out = self.unpool3(out)
        out = self.reflec_pad18(out)
        out = self.conv18(out)
        out = self.relu18(out)
        out = self.reflec_pad19(out)
        out = self.conv19(out)
        return out
