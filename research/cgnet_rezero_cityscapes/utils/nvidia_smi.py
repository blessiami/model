import subprocess
from time import sleep


class NvidiaSmi(object):
    def __init__(self, mem_thr=12188, act_thr=2000, period=1, show=True):
        super(NvidiaSmi, self).__init__()
        self._mem_thr = mem_thr
        self._act_thr = act_thr
        self._period = period
        self._show = show

    def get(self):
        sp = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        out = sp.communicate()
        msgs = out[0].decode("utf-8")

        if self._show:
            print(msgs)

        msgs = msgs.split()

        mems = []
        for msg in msgs:
            if msg.find("MiB") is not -1:
                mem = int(msg[:msg.find('MiB')])

                if mem < self._mem_thr:
                    mems.append(mem)

        return mems

    def chance(self):
        while True:
            try:
                cond = True
                mems = self.get()

                for mem in mems:
                    if mem > self._act_thr:
                        cond = False

                if cond:
                    break

            except KeyboardInterrupt:
                break

            sleep(self._period)


if __name__ == '__main__':
    nvidia_smi = NvidiaSmi()
    nvidia_smi.chance()
