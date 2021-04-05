---
layout: post
title: Linux Knowledge Tree
date: 2021-02-07 00:45 -0500
categories: [coding]
description: >
  Knowledge tree of Linux
image:
  path: "/assets/img/blog/linux.jpg"
related_posts: []
---

This winter break is one of the 2 breaks that I didn’t back to China in 7 years. I have some extra long time sitting in front of my computer. I think this is a good time for me to spend some effort to have a better knowledge of Linux. The learning resource I mainly rely on is [鸟哥的 Linux 私房菜](http://cn.linux.vbird.org/linux_basic/linux_basic.php). And my note will also reference the website.

* toc
{:toc .large-only}


### Computer Architecture
1. RISC and CISC: RISC (e.g. ARM) contains reduced small instructions and CISC (x86) contains multiple low level instructions.
2. NorthBridge connect to: CPU, RAM, GPU
SouthBridge connect to: Hard drive, USB, Internet card
3. Clock multiplier: Internal CPU clock rate / external supplied clock rate. Usually fixed. Overclocking is increase the external clock rate, so that internal CPU clock rate increase
4. 32-bit and 64-bit: The length of the RAM address that the CPU can handle. Each address corresponding to a byte, so that 32-bit machine can only have 4GB at most.
5. Dual Channel RAM: So that the RAM Bandwidth is 128-bit. Notice that the RAM also will be partitioned into 2 part. And each memory operations will be read/write to both partition.
6. DRAM and SRAM: DRAM is for RAM while SRAM is for L2 cache
7. Hard Drive: Sector (512bytes), Track, Cylinder
8. Operation System roles:
	1. System call interface
	2. Process control 
	3. Memory management
	4. FileSystem management
	5. Device driver


### Linux Intro
1. UNIX: Everything is a file
2. BSD: Berkeley Software Distribution, FreeBSD is a distribution of BSD
3. GNU: GNU is Not UNIX. Free software running on UNIX. Emacs, GCC & GLIBC, Bash shell
4. GPL: GNU General Public Liscense. [Comparison of free and open-source software licences - Wikipedia](https://en.wikipedia.org/wiki/Comparison_of_free_and_open-source_software_licences)
5. POSIX: Portable Operating System Interface. Interface between OS kernel and application
6. Linux: Unix-like system which based on POSIX. Version number: Odd number it tip branch Even number is stable branch.

### Linux Install
1. Disk Partition: First sector in hard disk contains 2 data: MBR (Master Boot Record) 446bytes and Partition Table 64bytes. 
2. Primary Partition, Extended Partition, Logical Partition:  We can have maximum 4 Primary + Extended. We can only have 1 Extended. And the Extended partition can be split to Logical Partitions (/dev/hda[] index start from 5). SATA hard drive can only have 11 Logical Partition to the most. 
3. Boot sequence: BIOS -> MBR -> Boot loader -> OS kernel
4. Boot Loader: installed on MBR by OS
	1. Provide boot options menu
	2. Load kernel files from the boot sector in the same partition
	3. Redirect to other boot loader (Windows) if needed
5. mount: mount a partition to the OS. Different folder under the same directory tree can be mounted by different partitions. Manually mount will be load to /mnt/myFile
6. Grub is the default boot loader for most distribution
7. Swap: Allocate disk space for temporary store less frequent used RAM data. 
8. ext4: Linux journaling file system
9. switch tty and GUI: [Ctrl] + [Alt] + [F1]~[F7]
10. runlevel: set OS running mode, e.g. with GUI or not

### File Permission
1. Overview: [Linux File Properties](http://cn.linux.vbird.org/linux_basic/0210filepermission_2.php)
2. Command for updating file permission:
	1. chgrp: change file user group
	2. chown: change the owner of the file
	3. chmod: change the permission of the file
Only the user currently has the permission of ‘w’ can exec the above commands
3. When assign permission to new user, we should be very careful for the ‘w’ permission
4. File type: [Linux File Types and Extensions](http://cn.linux.vbird.org/linux_basic/0210filepermission_2.php)
5. Linux File Hierarchy Standard: 
	1. /: root directory
	2. /usr: UNIX software resource, store software
	3. /var: store file for runtime
	4. /media and /mnt: for mount external storage devices. /media for long term mount and /mnt for temporary mount
	5. /proc: virtual filesystem that map system memory data into file system
	6. /etc, /bin, /dev, /lib, /sbin: these 5 folders must be in the same disk partition as root directory
	7. /usr/local: user installed software 
	8. Others: [Linux FHS](http://cn.linux.vbird.org/linux_basic/0210filepermission_3.php) 
6. SUID, SGID, SBIT: When running an executable that own by others and have set the SUID permission, the user is temporary grant the same permission as the owner. E.g. any user can execute passwd	

### Linux File System
1. Hard drive basics: [Hard drive basics](http://cn.linux.vbird.org/linux_basic/0230filesystem_1.php)
2. Traditional file system and software raid:
	1. Traditional File system: each partition can only be formatted to 1 file system
	2. Software Raid: 1 partition can be divided into multiple file system and multiple partition can together form a file system
3. Indexed Allocation (EXT2)
	1. Super block: record the overall info of the file system
	2. inode: Each inode record the indexing of block for a file, 1 inode can map to multiple blocks. 128bytes
	3. block: minimum storage unit for the file system, size can be 1KB/2KB/4KB. Block can also be used as indirect extension when storing big file
4. FAT (File Allocation Table): used by usb drive
	1. No inode
	2. block connection are stored within block like  linklist
	3. Fragmentation: the blocks for a file can be very scatter. So need to rearrange the blocks
5. Journaling File System (EXT3): Every time when need to update a file, first write the log to the journal system, so that if the actual write fail, we can always recover from log	
6. When create a new folder, we will be assign an inode and 1 block, that is why usually folder size is 4096. Also if too much file, might contain more blocks
7. Virtual FileSystem Switch: an interface layer between the OS and the actual file system. So that Linux can support multiple types of File system.
8. Link:
	1. Hard Link: file name link to the same inode
	2. Symbolic Link: file name link to different inodes
9. Swap: Use disk partition as RAM to store temporary not used memory data. 
	
### Compress File
1. Compress tool: 
	1. gzip: extension is .gz
	2. bzip2: extension is .bz2
	3. tar: modern compress tool
2. Backup

### vim
1. [vim](http://cn.linux.vbird.org/linux_basic/0310vi.php)

### bash
1. We can defined what type of shell to be launch for different users in /etc/passwd
2. Sub-shell cannot use the variables defined in parent shell. But they can be used after export to env
3. When assign variable: 
	1. “$VAR” will use the actual value of VAR
	2. ‘$VAR’ is just $VAR, ‘ means escape all
	3. Also if running command, the content in ‘’ will be run first: ls -l ‘locate crontab’
4. bash has 7 ttys, one of them is GUI
5. login shell and non-login shell
	1. login shell: the first shell that require login
	2. non-login shell: the shell launched after login shell. No need to login, but will not source /etc/profile
6. Sequence of sourcing variables: [bash](http://cn.linux.vbird.org/linux_basic/0320bash_4.php)
7. Bash special char: [bash](http://cn.linux.vbird.org/linux_basic/0320bash_4.php)
8. Data redirection, will redirect such channel into file, etc.
	1. stdin: 0
	2. stdout: 1
	3. stderr: 2
9. Pipe:
	1. Only redirect the stdout, ignore stderr
	2. The receiver command must able to receive pipe input

### Regex
1. Wildcard and regex: wildcard is an bash interface that used by regex
2. [regular expression](http://cn.linux.vbird.org/linux_basic/0330regularex.php)

### Shell script
1. #!/bin/bash
2. Different way of executing shell script
	1. ./script: run the script under a sub shell process, the variable will not be export
	2. source script: run in parent shell process
3. [Shell Scripts](http://cn.linux.vbird.org/linux_basic/0340bashshell-scripts.php)
4. Shell parameters: 
	1. $#: number of arguments
	2. $@:
	3. $*: concatenate arguments to string
	4. shift: shift argument index
5. Functions: usually running in parent shell process, unless:
	1. Run in background with &
	2. Run in a pipe
6. Static syntax check: sh -n script.sh
7. Verbose run: sh -x script.sh

### Account management
1. UID
	1. root user: 0
	2. system account: 1 - 499
	3. user: 500 - 65535
2. su: change account. su -: root
3. sudo: grant running root command to common user. Need to set account in /etc/sudoer
4. …

### Crontab
1. Set recurrent tasks for server machine, when get killed, will restart later
2. at: execute once
	1. ate/atrm: remove scheduled task
	2. batch: execute only when OS is not busy
3. anacron: handle recurrent task when machine is not always on.  

### Process and SELinux
1. UID: user ID. GID: user group ID
2. Thread ID and Process ID:
	1. PID is generated through fork, OS determine permission grant via PID 
	2. TID(PID) is generated within a process. It will also be assigned a PID. Thread and process are treated equally in linux. 
	3. PPID: parent PID. 
	4. TGID: PID of the first thread that created in a process
	5. bash PID is the PID for the started process
	6. [PID](https://stackoverflow.com/questions/9305992/if-threads-share-the-same-pid-how-can-they-be-identified)
3. fork-and-exec: when fork will generate a temporary process exactly as the parent. The only different is the PID and the PPID. Then exec the actual program
4. The first process of Linux OS PID is 1
5. daemon: service running in background
	1. Most daemon start script is at /etc/init.d/
6. job control
	1. &: throw job to background
	2. jobs: list background job number
	3. fg: bring background job to front
	4. bg: run the paused background job
	5. kill: -9 kill abnormal job, -15 normal stop a job
	6. kill -9 %jobnumber
	7. killall: kill all process under a parent process
	8. nohup: run task when log out
7. Signals:
	1. SIGHUP: restart
	2. SIGINT: ctrl+c
	3. SIGKILL: kill
	4. SIGTERM: normally stop
	5. SIGSTOP: ctrl+z pause
8. CPU scheduler:
	1. PRI: dynamic priority that only can be updated by OS, the lower the higher priority
	2. PRI = PRI + nice
	3. nice can be updated by user. 
		1. Root can set nice from -20 ~ 19
		2. user can set nice from 0 - 19 and user can only set nice more and more higher
		3. command: nice and renice 
9. /proc: map the OS related memory into file
10. SELinux:
	1. Subject: process
	2. Object: resource that need to be access
	3. Policy: policy
11. Enforce: restrict. Permissive: only show warning
	1. setenforce
12. [SELinux](http://cn.linux.vbird.org/linux_basic/0440processcontrol_5.php)

### Kernel Compile
1. [Linux compile](http://cn.linux.vbird.org/linux_basic/0540kernel.php)


---
### Useful Command
0. [Command List](http://cn.linux.vbird.org/linux_basic/1010index.php), [Command Cheatsheet](https://www.tecmint.com/linux-commands-cheat-sheet/)
1. startx: start X window from tty1-6
2. bc: calculator
3. Set Locale: LANG=en_US.UTF-8
4. sync: dump data in RAM to disk
5. Shutdown: shutdown -h now
6. su - [username] : change account
7. basename/dirname: obtain the file name or path
8. Print file content:
	1. cat/tac
	2. nl
	3. more
	4. less
	5. head
	6. tail
	7. od: read the file in binary format
9. umask: remove the default permission for a created file
10. file: print meta data of a file
11. Find files:
	1. which: find executable in PATH
	2. whereis: find file 
	3. locate: find file
	4. find: find file, slow
	5. type: find file
12. File system:
	1. df: read from superblock
	2. du: read from actual file, slow
13. link:
	1. ln: hard link
	2. ln -s: Symbolic Link
14. disk:
	1. fdisk: edit disk partition
	2. mkfs/mke2fs: format disk 
	3. fsck/badblocks: disk check
	4. mount: mount a media/partition to a specific folder
	5. umount: remove media device
	6. Other command for setting the disk parameters: mkond, e2label, tune2fs, heparm
15. Manage the boot default mount medias: Edit /etc/fstab 
16. dd: create an empty file. Also can copy files from disk sector
17. free: display memory usage
18. Compress:
	1. compress/gzip/zcat/bzip2/bzcat: mainly for compressing a single file
	2. tar: 
		1. compress: tar -jcv -f filename.tar.bz2
		2. check: tar -jtv -f filename.tar.bz2
		3. extract: tar -jxv -f filename.tar.bz2
19. Backup:
	1. dump: back file system
	2. restore: restore file system
	3. cpis: backup tool that work with find command
20. CD/DVD write:
	1. mkisofs: create iso file for content
	2. cdrecord: disk write tool
21. Env variables:
	1. env: get default variables
	2. set: get custom variables under bash
	3. unset: release a variable
	4. PS1: set the primary prompt variable
22. Reserved variable:
	1. echo $$: print PID
	2. echo $?: print last command result
	3. !!: exec last command
23. locale: set the language and text encoding
24. Variables:
	1. read: read keyboard input to a variable
	2. array: declare an array
	3. declare/typeset: declare variable
25. ulimit: set system usage limit
	1. ulimit -c unlimited: enable core dump
26. Modify Var:
	1. #, ##, %, %%, /, //
27. username=${username-root}: if username not set then set to root
28. alias rm=‘rm -i’
29. source == .
30. cat a.log >> b: no override original b
31. cmd > /dev/null 2>&1: redirect everything
	1. 2> will redirect the stderr
	2. &1 will redirect the stdout
32. sort related:
	1. sort
	2. wc: count
	3. uniq: count unique
33. tee: direct data to both screen and to file
34. char update command: tr, col, join, pastes, cut, expand
35. split: split file
36. xargs: when piping, create input pipe for next stage, incase some command cannot take pipe
37. -: single dash means waiting for stdin
38. --: double dash means any arguments after the -- are treated as filenames and arguments  
39. Text editing
	1. nl: number of lines
	2. sed: handling per line
	3. awk: handler base on delimiter
40. egrep: support extended regex format
41. diff tool:
	1. diff: compare per line
	2. cpp: compare per byte
	3. patch: update the file with diff patch
42. process:
	1. ps aux: check all process
	2. ps axjf: check process tree
	3. ps -l: only check bash process
	4. pstree
	5. top:
		1. -d: set interval  
		2. 1: display all cores
		3. i: only display active tasks
		4. M: rank by memory usage
43. free -m: check memory usage
44. Clean up memory
	1. Clear page check only: sync; echo 1 > /proc/sys/vm/drop_caches
	2. Clear inodes: sync; echo 2 > /proc/sys/vm/drop_caches
	3. Clear all: sync; echo 3 > /proc/sys/vm/drop_caches 
45. uname: check kernel info
46. dmesg: check kernel generated info
47. uptime: check start up time
48. netstat: network info
49. fuser: check the program that currently using the file
50. isof: list the file that opened by a process
51. ldd: list the shared lib that used by a program
  





















