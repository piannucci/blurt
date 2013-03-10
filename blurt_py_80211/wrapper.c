#include <unistd.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

int main( int argc, char ** argv, char ** envp )
{
    dup2(1, 2);
    if( setgid(getegid()) ) perror( "setgid" );
    if( setuid(geteuid()) ) perror( "setuid" );
    char *envp2[] = {
        "PATH=/opt/local/bin:/opt/local/sbin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/usr/local/bin:/opt/local/bin:/Users/peteriannucci/bin",
        "HOME=/Users/peteriannucci",
        "LANG=en_US.UTF-8",
        "__CF_USER_TEXT_ENCODING=0x1F5:0:0",
        "PYTHONPATH=/Users/peteriannucci/bin:",
        "SHELL=/bin/bash",
        "PKG_CONFIG_PATH=/usr/lib/pkgconfig:/usr/local/lib/pkgconfig:/opt/local/lib/pkgconfig"
    };
    fflush(stdout);

    if (argv[1] == 0)
    {
        printf("You must specify a WAV file to decode.\n");
        exit(1);
    }
    execle("/Users/peteriannucci/blurt/blurt_py_80211/blurt.py", "/Users/peteriannucci/blurt/blurt_py_80211/blurt.py", "--wav", argv[1], 0, envp2);
}
