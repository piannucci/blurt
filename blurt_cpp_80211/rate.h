#include "blurt.h"
#include "constellation.h"
#include "puncture.h"

class OFDMFormat;

class Rate {
public:
    uint32_t encoding;
    Constellation constellation;
    PuncturingMask puncturingMask;
    size_t Nbpsc, Nbps, Ncbps;
    Rate(uint32_t encoding, const Constellation &constellation, const PuncturingMask &puncturingMask, const OFDMFormat &format);
};
