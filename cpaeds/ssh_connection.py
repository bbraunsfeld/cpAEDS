import paramiko
import getpass

"""
Implements a class to handle a ssh connection build upon paramiko
"""

class SSHConnection():
    def __init__(self, host=None, user=None, password=None, encoding="UTF-8") -> None:
        """
        If fields are left empty, the user is prompted on runtime.

        host: hostname of the ssh host
        user: the username for the ssh connection
        password: the password for the ssh connection

        Example usage:

        server = SSHConnection() # This will query the user for host, user and password
        server.exec_command("echo 'hi'", path="/home/myuser/greetings/")
        server.closeSession()
        """
        self.encoding = encoding

        if host == None:
            self.host = input("Enter SSH host: ")
        else:
            self.host = host
        
        if user == None:
            self.user = input("Enter username for ssh connection: ")
        else:
            self.user = user

        if password == None:
            self.password = getpass.getpass("Enter ssh password (echo off):")
        else:
            self.password = password

        self.ssh_client = self.connect()

        return None
    
    def connect(self):
        ssh_client = paramiko.SSHClient()
        ssh_client.load_system_host_keys()
        ssh_client.set_missing_host_key_policy(paramiko.RejectPolicy()) # Don't connect to unknown hosts

        try:
            ssh_client.connect(hostname=self.host, username=self.user, password=self.password)
            return ssh_client

        except Exception as e:
            print(f"Could not connect to host: {e}")
            return False

    def exec_command(self, command, path=""):
        """
        Executes a command on the remote ssh connection.

        Returns: (stdout, stderr)

        command: the command to execute
        path: the path to execute the command at. If left empty, the command will be executed in the users home
        """
        stdin, stdout, stderr = self.ssh_client.exec_command(f"cd {path}; {command}")

        stdout_str = stdout.read().decode(self.encoding)
        stderr_str = stderr.read().decode(self.encoding)

        stdin.close()

        return (stdout_str, stderr_str)

    def closeSession(self):
        """
        Closes the ssh session.
        """

        self.ssh_client.close()
        return None