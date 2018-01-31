def calculate_label_confusion(matrix,label):
    tp = matrix[:,label,label]
    fn = matrix[:,label,:].sum(axis=1) - tp
    fp = matrix[:,:,label].sum(axis=1) - tp
    tn = matrix[:,:,:].sum(axis=(1,2)) - (fn+fp+tp)
    return tp,tn,fp,fn

def precision(tp,tn,fp,fn):
    return tp/(tp+fp)

def recall(tp,tn,fp,fn):
    return tp/(tp+fn)

def accuracy(tp,tn,fp,fn):
    return (tp+tn)/(tp+tn+fp+fn)