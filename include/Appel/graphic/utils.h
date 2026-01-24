#ifndef GRAPHIC_UTILS_H
#define GRAPHIC_UTILS_H

#include <Appel/graphic/frame.h>
#include <string>

namespace Appel {
  bool saveAsPng(const Frame &frame, const std::string &filename);
}

#endif
