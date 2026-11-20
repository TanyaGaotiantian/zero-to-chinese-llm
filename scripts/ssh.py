"""Tiny SSH/SFTP helper for the AutoDL instance (paramiko)."""
import sys, select, paramiko

HOST = "connect.cqa1.seetacloud.com"
PORT = 47873
USER = "root"
PASS = "DV37cNaz3nbW4bMKe+Oz5/Gt0oUfjGHI"


def client():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS,
              timeout=30, banner_timeout=30, auth_timeout=30)
    return c


def run(c, cmd, timeout=600, stream=True):
    ch = c.get_transport().open_session()
    ch.settimeout(timeout)
    ch.exec_command(cmd)
    out = []
    while True:
        r, _, _ = select.select([ch], [], [], timeout)
        if ch.recv_ready():
            d = ch.recv(65536).decode("utf-8", "replace")
            out.append(d)
            if stream:
                sys.stdout.write(d); sys.stdout.flush()
        if ch.recv_stderr_ready():
            d = ch.recv_stderr(65536).decode("utf-8", "replace")
            out.append(d)
            if stream:
                sys.stderr.write(d); sys.stderr.flush()
        if ch.exit_status_ready() and not ch.recv_ready() and not ch.recv_stderr_ready():
            break
    rc = ch.recv_exit_status()
    return rc, "".join(out)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "echo hello"
    c = client()
    rc, _ = run(c, cmd, timeout=int(sys.argv[2]) if len(sys.argv) > 2 else 900)
    c.close()
    sys.exit(rc)
