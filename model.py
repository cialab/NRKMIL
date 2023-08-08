import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class attmil(nn.Module):

    def __init__(self, inputd=1024, hd1=512, hd2=256):
        super(attmil, self).__init__()

        self.hd1 = hd1
        self.hd2 = hd2
        self.feature_extractor = nn.Sequential(
            nn.Linear(1024, hd1),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(hd1, hd2),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(hd1, hd2),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(hd2, 1)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=hd1, out_features=1)
        )


    def forward(self, x):
        x = self.feature_extractor(x) # mx512

        A_V = self.attention_V(x)  # mx256
        A_U = self.attention_U(x)  # mx256
        A = self.attention_weights(A_V * A_U) # element wise multiplication # mx1
        A = A.permute(0, 2, 1)  # 1xm
     
        #np.save("./CM16/test", A.cpu().detach().numpy()
        A = F.softmax(A, dim=2)  # softmax over m

        M = torch.matmul(A, x)  # 1x512
        M = M.view(-1, self.hd1) # 512

        Y_prob = self.classifier(M)

        return Y_prob, A
        
class attmil_ctranspath(nn.Module):

    def __init__(self, inputd=768, hd1=384, hd2=192):
        super(attmil_ctranspath, self).__init__()

        self.hd1 = hd1
        self.hd2 = hd2
        self.feature_extractor = nn.Sequential(
            nn.Linear(768, hd1),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(hd1, hd2),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(hd1, hd2),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(hd2, 1)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=hd1, out_features=1)
        )


    def forward(self, x):
        x = self.feature_extractor(x) # mx384

        A_V = self.attention_V(x)  # mx192
        A_U = self.attention_U(x)  # mx192
        A = self.attention_weights(A_V * A_U) # element wise multiplication # mx1
        A = A.permute(0, 2, 1)  # 1xm
     
        #np.save("./CM16/test", A.cpu().detach().numpy()
        A = F.softmax(A, dim=2)  # softmax over m

        M = torch.matmul(A, x)
        M = M.view(-1, self.hd1)

        Y_prob = self.classifier(M)

        return Y_prob, A
