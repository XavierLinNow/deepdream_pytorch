def make_step(X, model, **kwargs):
    #     X = X.copy()
    #   X是[1,c,w,h]的np array
    mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
    std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])

    learning_rate = kwargs.pop('lr', 5.0)
    max_jitter = kwargs.pop('max_jitter', 100)
    num_iterations = kwargs.pop('num_iterations', 100)
    show_every = kwargs.pop('show_every', 25)

    print(min, max)
    for t in range(num_iterations):
        ox, oy = np.random.randint(-max_jitter, max_jitter + 1, 2)
        X = np.roll(np.roll(X, ox, -1), oy, -2)

        model.zero_grad()
        X_tensor = torch.Tensor(X)
        X_Variable = Variable(X_tensor.cuda(), requires_grad=True)

        #         act_value = model(X_Variable)
        act_value = model.forward(X_Variable, 2)
        #         print(act_value.size())
        #         a = [0.]*1000
        #         a[111] = 1.
        #         a = torch.FloatTensor([a]).cuda()
        #         act_value.backward(a)
        act_value.backward(act_value.data)

        #         print('lr: {}'.format(learning_rate))
        learning_rate_ = learning_rate / np.abs(X_Variable.grad.data.cpu().numpy()).mean()
        #         print(learning_rate_)
        #         print(X_Variable.grad)
        X_Variable.data.add_(X_Variable.grad.data * learning_rate_)
        X = X_Variable.data.cpu().numpy()
        X = np.roll(np.roll(X, -ox, -1), -oy, -2)
        X[0, :, :, :] = np.clip(X[0, :, :, :], -mean / std, (1. - mean) / std)
        #         X[0,:,:,:] = np.clip(X[0,:,:,:], -0.5, 0.6)
        if t == 0 or (t + 1) % show_every == 0:
            inp = X[0, :, :, :]
            inp = inp.transpose(1, 2, 0)
            inp = std.reshape([1, 1, 3]) * inp + mean.reshape([1, 1, 3])
            inp *= 255
            showarray(inp)

    return X


def deepdream(model, base_img, lr, iter_n=10, octave_n=4, octave_scale=1.4,
              end='', clip=True, **step_params):
    mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
    std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])
    octaves = [base_img]
    for i in range(octave_n - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1, 1.0 / octave_scale, 1.0 / octave_scale), order=1))

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1, 1.0 * h / h1, 1.0 * w / w1), order=1)

        input_oct = octave_base + detail
        print(input_oct.shape)
        out = make_step(input_oct, model, lr=lr, num_iterations=iter_n, show_every=20)
        detail = out - octave_base
        print((detail.shape))
        #         inp = detail[0,:,:,:]
        #         inp = inp.transpose(1, 2, 0)
        #         inp = std.reshape([1,1,3]) * inp + mean.reshape([1,1,3])
        #         inp *= 255
        #         showarray(inp)