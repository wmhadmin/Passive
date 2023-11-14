import nltk



def isPassive(sentence):
    beforms = ['am', 'is', 'are', 'been', 'was', 'were', 'be', 'being']                                # all forms of "be"
    aux = ['do', 'did', 'does', 'have', 'has', 'had']                                                       # NLTK tags "do" and "have" as verbs, which can be misleading in the following section.
    words = nltk.word_tokenize(sentence)
    tokens = nltk.pos_tag(words)
    tags = [i[1] for i in tokens]
    if tags.count('VBN') == 0:                                                                                   # no PP, no passive voice.
        return False
    elif tags.count('VBN') == 1 and 'been' in words:                                                        # one PP "been", still no passive voice.
        return False
    else:
        pos = [i for i in range(len(tags)) if tags[i] == 'VBN' and words[i] != 'been']                            # gather all the PPs that are not "been".
        for end in pos:
            chunk = tags[:end]
            start = 0
            for i in range(len(chunk), 0, -1):
                last = chunk.pop()
                if last == 'NN' or last == 'PRP':
                    start = i                                                                                 # get the chunk between PP and the previous NN or PRP (which in most cases are subjects)
                    break
            sentchunk = words[start:end]
            tagschunk = tags[start:end]
            verbspos = [i for i in range(len(tagschunk)) if tagschunk[i].startswith('V')]                # get all the verbs in between
            if verbspos != []:                                                                              # if there are no verbs in between, it's not passive
                for i in verbspos:
                    if sentchunk[i].lower() not in beforms and sentchunk[i].lower() not in aux:         # check if they are all forms of "be" or auxiliaries such as "do" or "have".
                        break
                else:
                    return True
    return False


if __name__ == '__main__':

    samples = '''First-principles computations are making major inroads in advancing structure-property relations and providing insights into in silico prediction of materials. This model-driven approach has, however, had limited success for heterogeneous materials consisting of multiple surface sites, such as nanoparticles of various sizes and shapes, and/or of spatially inhomogeneously distributed elements, such as bimetallics and high entropy alloys. Solid catalysts used for electrochemical and thermochemical transformations, ranging from fuel cells to production of fuels and chemicals, are an important class of these materials. Compounding this difficulty in understanding, materials change dynamically in response to their environment, and thus, their ex situ characterization is often of limited value. The total number of surface sites of many metals is often quantified by CO chemisorption. Calorimetry and temperature programmed desorption (TPD) provide indirect and rather coarse-grained surface characterization1,2. Thus, the type of sites, as well their prevalence and contribution to catalyst performance remain largely unknown. First-principles models most often consider only one active site on a crystallographic plane, closely mimicking ultra-high vacuum single crystal experiments. The disparity between single crystal experiments, and associated calculations, and real-world materials is known as the materials gap3,4,5. Our ability to close the materials gap demands methods to quantify the types of surface sites of complex materials, along with their dynamic behavior under operando (working) conditions. This is an active research direction of both government funding agencies and private companies6,7.

Toward characterizing the structure of real nanomaterials, reverse Monte Carlo analysis of X-ray absorption fine structure spectroscopy (EXAFS) data has proven successful in resolving the structure of bimetallic nanoparticles with atomic resolution8. Neural networks, trained on X-ray absorption near edge structure (XANES) data, predict average coordination numbers of coordination shells9 and radial distribution functions10 given experimental spectra from monometallic nanoparticles. XAS spectroscopy is, though, a bulk technique11,12, whereas many phenomena, such as catalysis, depend directly on surface properties. Adsorbate vibrational excitations are, on the other hand, selective to adsorbate/surface interactions11. Infrared (IR) spectroscopy is commonly employed for characterizing adsorbate/surface, gas, and liquid-phase vibrational transitions. Fourier Transform IR (FTIR) spectroscopy with femtosecond time resolution can infer the structure of electronically excited transition metal complexes13, while two-dimensional spectroscopy tracks chemical transition states14 and coupling between vibrational modes15 in liquids. For solid surfaces, broad spectrum IR with nanometer spatial resolution is possible16. A major advantage of IR is that the spectra are very accurate17, capture details of most vibrational modes18,19 including coverage effects2,20,21, and can be obtained quickly in situ or operando for many environments22,23. Most IR-based peak assignments are heuristic and can be applied only to relatively simple spectra. Recent advances in IR-based quantification involve site-specific extinction coefficients24 in conjunction with peak deconvolution, integration, and a priori assumptions about particles sizes and adsorbate coverage distribution23. Due to the expense of first-principles calculations25, their direct use for detailed site and coverage identification would require generation and computation on a combinatorial number of structures to match spectra; this random match is beyond current and future computational power.

Here, we introduce a first-principles quantitative surface-selective IR methodology and integrate it with data-based approaches, chemistry-dependent problem formulation, and experimental data toward closing the materials gap to predict surface sites with atomic resolution from experimental data. Throughout the rest of this paper we refer to chemistry-dependent problem formulation and application of relevant physics as expert knowledge. We quantify error in both C–O and Pt–C frequencies of chemisorbed CO on Pt nanoparticles and extended surfaces and discover that density functional theory (DFT) generated spectra, even at high coverage, are much more accurate for determining adsorption site and deducing local microstructure than DFT energies. Our method untangles site-specific molecule/surface interactions, interprets complex experimental IR spectra, and quantitatively infers type and number of surface sites and adsorbate coverage. DFT-computed frequencies and intensities at low CO coverage serve as primary data; we feed this data through layers of physics-driven surrogate models to generate a secondary dataset of synthetic IR spectra to describe an arbitrary combination of adsorption sites and ultimately deduce structure. For each set of DFT frequencies and intensities, we quantify adsorption sites using both the binding-type (atop, bridge, threefold or fourfold) and the generalized coordination number (GCN)26; microstructure is then described using binding-type and GCN probability distribution functions (pdfs) from experimental IR data. We derive a closed-form solution for the derivative of the squared Wasserstein distance with respect to the softmax activation as a finite sum; this enables us to train two separate neural network ensembles to learn binding-type and GCN pdfs from synthetic spectra and quantify error in the predictions. Together, we refer to both neural network ensemble models as the structure surrogate model. We evaluate the structure surrogate model on both synthetic and experimental IR spectra, develop software, and implement the methodology with both CO and NO as probe molecules.'''




    sents = nltk.sent_tokenize(samples)
    totalCount = 0
    count = 0
    for sent in sents:
        totalCount = totalCount + 1
        if isPassive(sent) == True:
            count = count + 1

        print(sent + '--> %s' % isPassive(sent))


    print("总共句数：", totalCount, "    被动句",  count)