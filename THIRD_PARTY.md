# Dataset and inspiration

`data/input.txt` is the unmodified Tiny Shakespeare corpus distributed by
Andrej Karpathy in `char-rnn`:
https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare

Source download:
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

The underlying works are by William Shakespeare and are in the public domain.
The upstream `char-rnn` README identifies the project license as MIT. Credit
for compiling and distributing this training corpus belongs to Andrej Karpathy.
No upstream `char-rnn` training code is included in this workshop.

The bundled file contains 1,115,394 bytes. Its SHA-256 is:
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.

The experimental workflow is inspired by Andrej Karpathy's MIT-licensed
autoresearch project: https://github.com/karpathy/autoresearch . This is a
separate small CPU teaching implementation, not a port of its GPU training code.

PyTorch is a dependency with its own third-party licenses:
https://github.com/pytorch/pytorch/blob/main/LICENSE
