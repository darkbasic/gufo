#pragma once

#include <iosfwd>

namespace gufo::diagnostics {

// The native probe and its CPU test execute the same bounded phase sequence.
int RunHipResourceProbe(int argc, const char* const* argv, std::ostream& output,
                        std::ostream& errors);

}  // namespace gufo::diagnostics
