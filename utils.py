def encoder_decoder(classes):
    decoder = {}
    for i in range(len(classes)):
        decoder[classes[i]] = i

    encoder = {}
    for i in range(len(classes)):
        encoder[i] = classes[i]

    return encoder, decoder
