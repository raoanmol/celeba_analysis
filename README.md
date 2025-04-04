If you get `RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)`, a potential fix that worked for me was:
1. Go to the `captum/concept/_utils/classifier.py` file.
2. Go to line ~196 which says `predict = self.lm.classes()[torch.argmax(predict, dim=1)]`.
3. Change it to `predict = self.lm.classes()[torch.argmax(predict.cpu(), dim=1)]`.

This forces the index tensor to CPU which somehow fixes the device mismatch crash.