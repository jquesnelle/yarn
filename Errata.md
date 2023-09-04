## Mistakes in the Early Preprint
Thank you to every reader that pointed these mistakes out!

### Section 2.2
Equation 9: Terms `mL'/L` should be `mL/L'` instead.

### Section 2.3
Equation 12: `s*m` should be `m/s` instead.

### Section 3.4

This part's hypothesis is backwards (we mixed up and inverted the data by accident, and multiplied the logits by `t` instead of dividing)
```
[...], it skews the attention softmax distribution to become "spikier" (i.e.
decreases the average entropy of the attention softmax). [...] the network "pays more attention" to more tokens.
[...]
Since there is a decrease of entropy in the attention softmax distribution as we interpolate RoPE
embeddings to longer context sizes, our goal is to reverse the decrease of entropy (i.e. increase the
"temperature" of the attention logits).
```

It should say instead:
```
[...] it skews the attention softmax distribution to become "smoother" (i.e.
increases the average entropy of the attention softmax). [...] the network "pays less attention" to specific tokens.
[...]
Since there is an increase of entropy in the attention softmax distribution as we interpolate RoPE
embeddings to longer context sizes, our goal is to reverse the increase of entropy (i.e. decrease the
"temperature" of the attention logits).
```

In the same vein, all temperature variables should be written as `1/t` instead of `t`, as we divide logits by the temperature, not multiply.

However, Equation 27 still stands, except the left part should be `sqrt(1/t)` instead of `sqrt(t)`.
