#include <array>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#include "gufo/ext/generated/qwen38_flash_next_constants.hpp"
#include "src/models/qwen38_flash_next/runner.hpp"

namespace qfn = gufo::models::qwen38_flash_next;

static_assert(qfn::DecodeConcurrencyForSessions(0) == 1);
static_assert(qfn::DecodeConcurrencyForSessions(1) == 1);
static_assert(qfn::DecodeConcurrencyForSessions(2) == 2);
static_assert(qfn::DecodeConcurrencyForSessions(4) == 4);
static_assert(qfn::DecodeConcurrencyForSessions(8) == 8);
static_assert(qfn::DecodeConcurrencyForSessions(9) == 8);
static_assert(qfn::DecodeConcurrencyForSessions(
                  std::numeric_limits<std::size_t>::max()) == 8);

namespace {
void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

void RejectedVisionSettings(const gufo::ext::TextRunnerOptions& options,
                            std::string_view diagnostic) {
  for (const bool budget_helper : {false, true}) {
    bool rejected = false;
    try {
      if (budget_helper)
        (void)qfn::VisionMemoryBudgetBytes(options);
      else
        (void)qfn::VisionMemoryOnDemand(options);
    } catch (const std::invalid_argument& error) {
      rejected = std::string_view(error.what()).find(diagnostic) !=
                 std::string_view::npos;
    }
    Expect(rejected, "both option helpers reject invalid vision settings");
  }
}
}  // namespace

int main() {
  // The metadata is generated from the real provider declaration. Simulate
  // HIP availability without loading the provider or touching a device.
  gufo::ext::Registry registry;
  registry.BeginExtension({"qwen38_flash_next", "test", ""});
  auto provider = gufo::ext::generated::extension_qwen38_flash_next::
      kProvider_flash_next_q4;
  provider.available = true;
  provider.unavailable_reason.clear();
  registry.AddProvider(std::move(provider));
  gufo::ext::ProviderInfo unrelated;
  unrelated.id = "unrelated-provider";
  registry.AddProvider(std::move(unrelated));
  const std::array<std::string_view, 2> arguments{"--vision-memory-bytes",
                                                  "4294967296"};
  auto parsed = gufo::ext::ParseOptions(registry, "serve", arguments);
  gufo::ext::ValidateSelectedOptions(
      registry, "serve", registry.Provider("flash-next-q4"), parsed);
  gufo::ext::TextRunnerOptions options;
  Expect(qfn::VisionMemoryBudgetBytes(options) == 0,
         "missing option preserves conservative admission");
  Expect(!qfn::VisionMemoryOnDemand(options),
         "on-demand is never selected by default");
  options.provider_options["--vision-memory-bytes"] = std::int64_t{0};
  Expect(qfn::VisionMemoryBudgetBytes(options) == 0,
         "explicit zero restores conservative admission without a projector");
  Expect(!qfn::VisionMemoryOnDemand(options),
         "legacy zero is not reinterpreted as on-demand");
  options.vision_model_path = "projector.gguf";
  Expect(qfn::VisionMemoryBudgetBytes(options) == 0,
         "projector presence does not opt into request-aware admission");
  options.provider_options = parsed;
  Expect(qfn::VisionMemoryBudgetBytes(options) == std::size_t{4294967296},
         "positive option preserves the full configured byte count");
  Expect(!qfn::VisionMemoryOnDemand(options),
         "legacy positive bytes retain the fixed-pool guarantee");
  options.provider_options["--vision-memory-bytes"] =
      std::numeric_limits<std::int64_t>::max();
  Expect(qfn::VisionMemoryBudgetBytes(options) ==
             static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()),
         "largest parsed integer is represented without narrowing");
  for (const auto value :
       {std::int64_t{-1}, std::numeric_limits<std::int64_t>::min()}) {
    options.provider_options["--vision-memory-bytes"] = value;
    bool rejected = false;
    try {
      (void)qfn::VisionMemoryBudgetBytes(options);
    } catch (const std::invalid_argument& error) {
      rejected = std::string_view(error.what()).find("--vision-memory-bytes") !=
                 std::string_view::npos;
    }
    Expect(rejected, "negative budgets fail with the option name");
  }
  options.vision_model_path.clear();
  options.provider_options["--vision-memory-bytes"] = std::int64_t{1};
  bool rejected = false;
  try {
    (void)qfn::VisionMemoryBudgetBytes(options);
  } catch (const std::invalid_argument& error) {
    rejected = std::string_view(error.what()).find("--mmproj") !=
               std::string_view::npos;
  }
  Expect(rejected, "request-aware admission requires a vision projector");
  for (const auto value : {"", "4GiB", "1.5", "18446744073709551616"}) {
    const std::array<std::string_view, 2> malformed{"--vision-memory-bytes",
                                                    value};
    rejected = false;
    try {
      (void)gufo::ext::ParseOptions(registry, "serve", malformed);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    Expect(rejected, "CLI rejects malformed and overflowing byte counts");
  }
  rejected = false;
  try {
    gufo::ext::ValidateSelectedOptions(
        registry, "serve", registry.Provider("unrelated-provider"), parsed);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Expect(rejected, "providers without request-aware support reject the option");
  rejected = false;
  try {
    (void)gufo::ext::ParseOptions(registry, "prompt", arguments);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Expect(rejected, "the opt-in is confined to the serving runner");

  const std::array<std::string_view, 2> on_demand_arguments{
      "--vision-memory-policy", "on-demand"};
  parsed = gufo::ext::ParseOptions(registry, "serve", on_demand_arguments);
  gufo::ext::ValidateSelectedOptions(
      registry, "serve", registry.Provider("flash-next-q4"), parsed);
  options.provider_options = parsed;
  options.vision_model_path = "projector.gguf";
  Expect(qfn::VisionMemoryOnDemand(options) &&
             qfn::VisionMemoryBudgetBytes(options) == 0,
         "explicit on-demand uses no unused fixed startup pool");
  options.vision_model_path.clear();
  RejectedVisionSettings(options, "--mmproj");
  options.vision_model_path = "projector.gguf";
  for (const auto value : {std::int64_t{0}, std::int64_t{1},
                           std::int64_t{4294967296}, std::int64_t{-1}}) {
    options.provider_options["--vision-memory-bytes"] = value;
    RejectedVisionSettings(options, "cannot be combined");
  }
  options.provider_options.erase("--vision-memory-bytes");
  for (const auto policy : {"", "automatic", "fixed", "ON-DEMAND"}) {
    options.provider_options["--vision-memory-policy"] = std::string(policy);
    RejectedVisionSettings(options, "conservative or on-demand");
  }

  options.provider_options["--vision-memory-policy"] =
      std::string("conservative");
  options.vision_model_path.clear();
  Expect(!qfn::VisionMemoryOnDemand(options) &&
             qfn::VisionMemoryBudgetBytes(options) == 0,
         "explicit conservative admission needs no vision projector");
  options.provider_options["--vision-memory-bytes"] = std::int64_t{0};
  Expect(!qfn::VisionMemoryOnDemand(options) &&
             qfn::VisionMemoryBudgetBytes(options) == 0,
         "conservative and legacy zero express the same guarantee");
  options.provider_options["--vision-memory-bytes"] = std::int64_t{1};
  RejectedVisionSettings(options, "cannot be combined");

  rejected = false;
  try {
    gufo::ext::ValidateSelectedOptions(
        registry, "serve", registry.Provider("unrelated-provider"), parsed);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Expect(rejected, "unsupported providers reject the on-demand policy");
  for (const auto command : {"prompt", "bench"}) {
    rejected = false;
    try {
      (void)gufo::ext::ParseOptions(registry, command, on_demand_arguments);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    Expect(rejected, "on-demand policy is confined to the serving runner");
  }
}
