import torch


class BCEAcc():
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.correct = 0
        self.total_count = 0

    def forward(self, x, y):
        bs = x.shape[0]
        x = torch.relu(torch.sign(x))
        x = (x==y).int().sum(dim=0)

        self.total_count += bs
        self.correct += x

        return x.sum().item()/bs/x.numel(), bs

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def stats(self, mode='primary'):
        if mode=='primary' : return self.correct.sum()/(self.total_count*self.correct.numel())
        else: return {'acc': self.correct/self.total_count}

    def db(self):
        return {
            'corrects':self.correct,
            'total_count':self.total_count,
        }

    def print_stats(self, mode='val'):
        data = self.stats(mode=mode)
        for i in data.items():
            print(i)


class BCEMed():
    '''BCEMed calculates count of true posetives, true negatives, total posetives, and total negatives 
    addition to accuracy calculated in BCEAccS'''
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.correct = 0
        self.tp = 0
        self.tn = 0
        self.pos = 0
        self.neg = 0
        
        self.total_count = 0

    def forward(self, x, y):
        bs = x.shape[0]
        x = torch.relu(torch.sign(x))
        x = (x==y).int()

        # sensetivity
        tp = (x*y).sum(dim=0)
        self.tp += tp
        pos = y.sum(dim=0)
        self.pos += pos

        # specificity
        neg = -y+1
        tn = (x*neg).sum(dim=0)
        self.tn += tn
        self.neg += neg.sum(dim=0)

        # overall accuracy
        acc = x.sum(dim=0)
        self.correct += acc

        self.total_count += bs
        return acc.sum().item()/bs/acc.numel(), bs

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def stats(self, mode='primary'):
        if mode=='primary':
            return self.correct.sum()/(self.total_count*self.correct.numel())
        elif mode=='trn':
            return {'acc_trn': self.correct.sum()/(self.total_count*self.correct.numel())}
        elif mode=='val':
            return self.acc_med()
        else: raise Exception

    def acc_med(self):
        overall = {
            'acc_val' : self.correct.sum()/(self.total_count*self.correct.numel()),
            'sensetivity': self.tp.sum()/self.pos.sum(),
            'specificity': self.tn.sum()/self.neg.sum(),
            'dice' : (self.tp*self.tn).sum()/((self.pos-self.tp)*(self.neg-self.tn)).sum(),

            'accs' : self.correct/(self.total_count*self.correct.numel()),
            'senss': self.tp/self.pos,
            'specs': self.tn/self.neg,
            'dices' : (self.tp*self.tn)/((self.pos-self.tp)*(self.neg-self.tn)),
        }
        return overall
        
    def db(self):
        return {
            'corrects':self.correct,
            'true_pos':self.tp,
            'true_neg':self.tn,
            'total_pos':self.pos,
            'total_neg':self.neg,
            'total_count':self.total_count,
        }


    def print_stats(self, mode='val'):
        data = self.stats(mode=mode)
        for i in data.items():
            print(i)


class SoftmaxCorrect():
    """docstring for Softmax_Acc"""
    def __init__(self, classes_out:int=None, device=None):
        super().__init__()
        if classes_out is not None: self.correct = torch.zeros(classes_out, device=device)
        else: self.correct = 0
        self.classes_out = classes_out

        self.correct = 0
        
        self.total_count = 0

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self, x, labels) -> float:
        bs = labels.shape[0]
        self.total_count += bs
        _, preds = torch.max(x, dim=1)

        if self.classes_out is not None:
            x = self.one_hot(preds)
            y = self.one_hot(labels)

            x = torch.relu(torch.sign(x))
            correct = (x==y).int().sum(dim=0)

        else:
            correct = (preds==labels).sum()

        self.correct += correct
        return correct/bs, bs
        
    def stats(self, mode='primary'):
        if mode=='primary':
            return self.correct/self.total_count
        else:
            return {f'acc_{mode}': self.correct/self.total_count}

    def print_stats(self, mode='val'):
        data = self.stats(mode=mode)
        for i in data.items():
            print(i)

    def one_hot(self, x):
        bs = x.shape[0]
        out = torch.zeros(bs, self.classes_out)
        out[torch.arange(bs), x] = 1.
        return out


class SoftmaxMed():
    """docstring for Softmax_Acc"""
    def __init__(self, classes_out, device=None):
        super().__init__()
        self.correct = torch.zeros(classes_out)
        self.tp = torch.zeros(classes_out)
        self.tn = torch.zeros(classes_out)
        self.pos = torch.zeros(classes_out)
        self.neg = torch.zeros(classes_out)
        
        self.total_count = 0
        self.classes_out = classes_out

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self, x, labels) -> float:
        bs = x.shape[0]
        self.total_count += bs

        _, preds = torch.max(x, dim=1)

        x = self.one_hot(preds)
        y = self.one_hot(labels)

        x = torch.relu(torch.sign(x))
        x = (x==y).int()

        # sensetivity
        tp = (x*y).sum(dim=0)
        self.tp += tp
        pos = y.sum(dim=0)
        self.pos += pos

        # specificity
        neg = -y+1
        tn = (x*neg).sum(dim=0)
        self.tn += tn
        self.neg += neg.sum(dim=0)

        # overall accuracy
        acc = x.sum(dim=0)
        self.correct += acc

        self.total_count += bs
        return tp.sum().item()/pos.sum().item()


    def one_hot(self, x):
        bs = x.shape[0]
        out = torch.zeros(bs, self.classes_out)
        out[torch.arange(bs), x] = 1.
        return out

    def acc_med(self):
        overall = {
            'acc_val' : self.tp.sum()/self.pos.sum(),
            'sensetivity': self.tp.sum()/self.pos.sum(),
            'specificity': self.tn.sum()/self.neg.sum(),
            'dice' : (self.tp*self.tn).sum()/((self.pos-self.tp)*(self.neg-self.tn)).sum(),

            'accs ' : self.correct/self.total_count,
            'senss': self.tp/self.pos,
            'specs': self.tn/self.neg,
            'dices' : (self.tp*self.tn)/((self.pos-self.tp)*(self.neg-self.tn)),
        }
        return overall
        
    def stats(self, mode='primary'):
        if mode=='primary':
            return self.tp.sum()/self.pos.sum()
        elif mode=='trn':
            return {'acc_trn': self.tp.sum()/self.pos.sum()}
        elif mode=='val':
            return self.acc_med()
        else: raise Exception

    def print_stats(self, mode='val'):
        data = self.stats(mode=mode)
        for i in data.items():
            print(i)


