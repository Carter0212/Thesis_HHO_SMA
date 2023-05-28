from dataclasses import dataclass,field
import dataclasses
import json
@dataclass
class jsonDATA:

    pin: list = field(default_factory=list) ## Pin PAL進入的數量
    pout: list = field(default_factory=list) ## pout PAL出去的數量
    gin :list = field(default_factory=list)## gin GAA進入的數量
    gout : list = field(default_factory=list) ## gout GAA出去的數量
    a : list = field(default_factory=list) ## a PAL 進入的數量list版
    b :list = field(default_factory=list) ## a PAL 進入的數量list版
    n :list = field(default_factory=list)
    np :list = field(default_factory=list)
    ##
    p:list = field(default_factory=list)
    g:list = field(default_factory=list)
    x:list = field(default_factory=list)
    y:list = field(default_factory=list)
    z:list = field(default_factory=list)
    a:list = field(default_factory=list)
    AACP:list = field(default_factory=list)
    NAACP:list = field(default_factory=list)
    b:list = field(default_factory=list)
    AACG:list = field(default_factory=list)
    NAACG:list = field(default_factory=list)
    asum:list = field(default_factory=list)
    bsum:list = field(default_factory=list)
    dx:list = field(default_factory=list)
    dy:list = field(default_factory=list)
    dgtg:list = field(default_factory=list)
    interf:list = field(default_factory=list)
    interfsum:list = field(default_factory=list)
    u : list = field(default_factory=list)
    v : list = field(default_factory=list)
    vsum : list = field(default_factory=list)
    CAC : list = field(default_factory=list)
    NCAC : list = field(default_factory=list)
    coordinate : list = field(default_factory=list)
    utitlity : list = field(default_factory=list)
    pal_movemont : list = field(default_factory=list)
    gaa_movemont : list = field(default_factory=list)
    gaa_no_allocat : list = field(default_factory=list)
    interfull : list = field(default_factory=list)
    uCAC : list = field(default_factory=list)
    uNCAC : list = field(default_factory=list)
    CA : list = field(default_factory=list)
    pal_no_allocat : list = field(default_factory=list)
    alpha_utitlity : list = field(default_factory=list)
    beta_utitlity : list = field(default_factory=list)

    def AppendData(self,nowTimedata,Now_env,old_env,countGNA,interfull,utitlity,countPNA,alpha_utitlity,beta_utitlity):
        ## have utility data
        self.CA.append(Now_env.CA)
        self.pin.append(nowTimedata.pin)
        self.pout.append(nowTimedata.pout)
        self.gin.append(nowTimedata.gin)
        self.gout.append(nowTimedata.gout)
        self.n.append(Now_env.n)
        self.np.append(Now_env.np)
        self.pal_movemont.append(sum([int(a)-int(b) for a,b in zip(Now_env.u,old_env.u)]))
        self.gaa_movemont.append(sum([int(a)-int(b) for a,b in zip(Now_env.v,old_env.v)]))
        # self.pal_movemont.append(sum(Now_env.u))
        # self.gaa_movemont.append(sum(Now_env.v))
        self.gaa_no_allocat.append(countGNA)
        self.interfull.append(interfull)
        self.utitlity.append(utitlity)

        ## no have utility data
        self.p.append(Now_env.p)
        self.g.append(Now_env.g)
        self.x.append(Now_env.x)
        self.y.append(Now_env.y)
        self.z.append(Now_env.z)
        self.a.append(Now_env.a)
        self.AACP.append(Now_env.AACP)
        self.NAACP.append(Now_env.NAACP)
        self.b.append(Now_env.b)
        self.AACG.append(Now_env.AACG)
        self.NAACG.append(Now_env.NAACG)
        self.asum.append(Now_env.asum)
        self.bsum.append(Now_env.bsum)
        self.dx.append(Now_env.dx)
        self.dy.append(Now_env.dy)
        self.dgtg.append(Now_env.dgtg)
        self.interf.append(Now_env.interf)
        self.interfsum.append(Now_env.interfsum)
        self.u.append(Now_env.u)
        self.uCAC.append(Now_env.uCAC)
        self.uNCAC.append(Now_env.uNCAC)
        self.v.append(Now_env.v)
        self.CAC.append(Now_env.CAC)
        self.NCAC.append(Now_env.NCAC)
        self.pal_no_allocat.append(countPNA)
        self.alpha_utitlity.append(alpha_utitlity)
        self.beta_utitlity.append(beta_utitlity)
    
    def ToDict(self):
        return dataclasses.asdict(self)

    # def GetAverage(self,addClass):
    #     self.pal_movemont

    ##def PreAppendData(self,nowTimedata,Now_env):
    
def DictToJson(Dict,savePath,Algname):
    with open(f'{savePath}{Algname}_json.json', 'w') as json_file:
        json.dump(Dict, json_file)

