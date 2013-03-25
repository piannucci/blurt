#include "blurt.h"
#include "constellation.h"
#include "puncture.h"

class OFDMFormat;

class Rate {
public:
    int encoding;
    Constellation constellation;
    PuncturingMask puncturingMask;
    int Nbpsc, Nbps, Ncbps;
    Rate(int encoding, const Constellation &constellation, const PuncturingMask &puncturingMask, const OFDMFormat &format);
};
