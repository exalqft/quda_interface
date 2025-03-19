#include <util_quda.h>
#include <host_utils.h>
#include <command_line_params.h>

#include <quda.h>

int main(int argc, char **argv)
{
  auto app = make_app();
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // initialize the QUDA library
  initQuda(device_ordinal);

  // end quda
  endQuda();

  // finalize comms
  finalizeComms();

  return 0;
}