"""
# ggrocket 
## A ggmath extensions for modeling spacecraft in planetary orbit
"""

from abc import ABCMeta, abstractmethod
from collections import namedtuple

from math import pi, degrees, radians, atan2, sin, cos, sqrt
from ggame import LineStyle, Color, ImageAsset
#from ggame.mathapp import MathApp
from ggame.circle import Circle
#from ggame.point import ImagePoint
from ggame.timer import Timer
from ggame.label import Label
from ggame import Sprite
from time import time


class _MathDynamic(metaclass=ABCMeta):
    
    def __init__(self):
        self._dynamic = False  # not switched on, by default!
    
    def destroy(self):
        MathApp._removeDynamic(self)

    def step(self):
        print("_MathDynamic step")
        pass
    
    def Eval(self, val):
        if callable(val):
            print("eval", val)
            self._setDynamic() # dynamically defined .. must step
            return val
        else:
            return lambda : val  
            
    def _setDynamic(self):
        MathApp._addDynamic(self)
        self._dynamic = True
            

class _MathVisual(Sprite, _MathDynamic, metaclass=ABCMeta):
    """
    Abstract Base Class for all visual, potentially dynamic objects.
    
    :param Asset asset: A valid ggame asset object.
    
    :param list args: A list of required positional or non-positional arguments
        as named in the _posinputsdef and _nonposinputsdef lists overridden
        by child classes.
        
    :param \**kwargs:
        See below

    :Optional Keyword Arguments:
        * **positioning** (*string*) One of 'logical' or 'physical'
        * **size** (*int*) Size of the object (in pixels)
        * **width** (*int*) Width of the object (in pixels)
        * **color** (*Color*) Valid :class:`~ggame.asset.Color` object
        * **style** (*LineStyle*) Valid :class:`~ggame.asset.LineStyle` object
    """
    
    _posinputsdef = []  # a list of names (string) of required positional inputs
    _nonposinputsdef = []  # a list of names (string) of required non positional inputs
    _defaultsize = 15
    _defaultwidth = 200
    _defaultcolor = Color(0, 1)
    _defaultstyle = LineStyle(1, Color(0, 1))
    
    
    def __init__(self, asset, *args, **kwargs):
        MathApp._addVisual(self)
        #Sprite.__init__(self, asset, args[0])
        _MathDynamic.__init__(self)
        self._movable = False
        self._selectable = False
        self._strokable = False
        self.selected = False
        """
        True if object is currently selected by the UI. 
        """
        self.mouseisdown = False
        """
        True if object is tracking UI mouse button as down. 
        """
        self._positioning = kwargs.get('positioning', 'logical')
        # positional inputs
        self._PI = namedtuple('PI', self._posinputsdef)
        # nonpositional inputs
        self._NPI = namedtuple('NPI', self._nonposinputsdef)
        # standard inputs (not positional)
        standardargs = ['size','width','color','style']
        self._SI = namedtuple('SI', standardargs)
        # correct number of args?
        if len(args) != len(self._posinputsdef) + len(self._nonposinputsdef):
            raise TypeError("Incorrect number of parameters provided")
        self._args = args
        # generated named tuple of functions from positional inputs
        self._posinputs = self._PI(*[self.Eval(p) for p in args][:len(self._posinputsdef)])
        self._getPhysicalInputs()
        # first positional argument must be a sprite position!
        Sprite.__init__(self, asset, self._pposinputs[0])
        # generated named tuple of functions from nonpositional inputs
        if len(self._nonposinputsdef) > 0:
            self._nposinputs = self._NPI(*[self.Eval(p) for p in args][(-1*len(self._nonposinputsdef)):])
        else:
            self._nposinputs = []
        self._stdinputs = self._SI(self.Eval(kwargs.get('size', self._defaultsize)),
                                    self.Eval(kwargs.get('width', self._defaultwidth)),
                                    self.Eval(kwargs.get('color', self._defaultcolor)),
                                    self.Eval(kwargs.get('style', self._defaultstyle)))
        self._sposinputs = self._PI(*[0]*len(self._posinputs))
        self._spposinputs = self._PI(*self._pposinputs)
        self._snposinputs = self._NPI(*[0]*len(self._nposinputs))
        self._sstdinputs = self._SI(*[0]*len(self._stdinputs))

    def step(self):
        print("_MathVisual step")
        self._touchAsset()
        
    def _saveInputs(self, inputs):
        self._sposinputs, self._spposinputs, self._snposinputs, self._sstdinputs = inputs
        
    def _getInputs(self):
        self._getPhysicalInputs()
        return (self._PI(*[p() for p in self._posinputs]),
            self._PI(*self._pposinputs),
            self._NPI(*[p() for p in self._nposinputs]),
            self._SI(*[p() for p in self._stdinputs]))

    
    def _getPhysicalInputs(self):
        """
        Translate all positional inputs to physical
        """
        pplist = []
        if self._positioning == 'logical':
            for p in self._posinputs:
                pval = p()
                try:
                    pp = MathApp.logicalToPhysical(pval)
                except AttributeError:
                    pp = MathApp._scale * pval
                pplist.append(pp)
        else:
            # already physical
            pplist = [p() for p in self._posinputs]
        self._pposinputs = self._PI(*pplist)
    
    def _inputsChanged(self, saved):
        return self._spposinputs != saved[1] or self._snposinputs != saved[2] or self._sstdinputs != saved[3]

    
    def destroy(self):
        MathApp._removeVisual(self)
        MathApp._removeMovable(self)
        MathApp._removeStrokable(self)
        _MathDynamic.destroy(self)
        Sprite.destroy(self)

    def _updateAsset(self, asset):
        if type(asset) != ImageAsset:
            visible = self.GFX.visible
            if App._win != None:
                App._win.remove(self.GFX)
                self.GFX.destroy()
            self.asset = asset
            self.GFX = self.asset.GFX
            self.GFX.visible = visible        
            if App._win != None:
                App._win.add(self.GFX)
        self.position = self._pposinputs.pos
            
    @property
    def positioning(self):
        """
        Whether object was created with 'logical' or 'physical' positioning. 
        """
        return self._positioning
    
    @positioning.setter
    def positioning(self, val):
        pass
    
    @property
    def movable(self):
        """
        Whether object can be moved. Set-able and get-able. 
        """
        return self._movable
        
    @movable.setter
    def movable(self, val):
        if not self._dynamic:
            self._movable = val
            if val:
                MathApp._addMovable(self)
            else:
                MathApp._removeMovable(self)

    @property
    def selectable(self):
        """
        Whether object can be selected by the UI. Set-able and get-able.
        """
        return self._selectable
        
    @selectable.setter
    def selectable(self, val):
        self._selectable = val
        if val:
            MathApp._addSelectable(self)
        else:
            MathApp._removeSelectable(self)

    @property
    def strokable(self):
        """
        Whether the object supports a click-drag input from the UI mouse. 
        Set-able and get-able. 
        """
        return self._strokable
        
    @strokable.setter
    def strokable(self, val):
        self._strokable = val
        if val:
            MathApp._addStrokable(self)
        else:
            MathApp._removeStrokable(self)

    def select(self):
        """
        Place the object in a 'selected' state. 
        
        :param: None
        :returns: None 
        """
        self.selected = True

    def unselect(self):
        """
        Place the object in an 'unselected' state. 
        
        :param: None
        :returns: None
        """
        self.selected = False
        
    def mousedown(self):
        """
        Inform the object of a 'mouse down' event. 
        
        :param: None 
        :returns: None 
        """
        self.mouseisdown = True
        
    def mouseup(self):
        """
        Inform the object of a 'mouse up' event. 
        
        :param: None 
        :returns: None 
        """
        self.mouseisdown = False

    def processEvent(self, event):
        """
        Inform the object of a generic ggame event. 
        
        :param event: The ggame event object to receive and process. 
        :returns: None 
        
        This method is intended to be overridden.
        """
        pass

    @abstractmethod
    def physicalPointTouching(self, ppos):
        """
        Determine if a physical point is considered to be touching this object.
        
        :param tuple(int,int) ppos: Physical screen coordinates.
        :rtype: boolean
        :returns: True if touching, False otherwise.
        
        This method **must** be overridden.
        """
        pass
    
    @abstractmethod
    def translate(self, pdisp):
        """ 
        Perform necessary processing in response to being moved by the mouse/UI.
        
        :param tuple(int,int) pdisp: Translation vector (x,y) in physical screen
            units.
        :returns: None
        
        This method **must** be overridden.
        """
        pass
    
    def stroke(self, ppos, pdisp):
        """
        Perform necessary processing in response to click-drag action by the
        mouse/UI.
        
        :param tuple(int,int) ppos: Physical coordinates of stroke start.
        :param tuple(int,int) pdisp: Translation vector of stroke action in
            physical screen units.
        :returns: None
        
        This method is intended to be overridden.
        """
        pass
    
    def canStroke(self, ppos):
        """
        Can the object respond to beginning a stroke action at the given
        position.
        
        :param tuple(int,int) ppos: Physical coordinates of stroke start.
        :rtype: Boolean
        :returns: True if the object can respond, False otherwise.
        
        This method is intended to be overridden.
        """
        return False
    
    def _touchAsset(self, force = False):
        inputs = self._getInputs()
        changed = self._inputsChanged(inputs)
        if changed:
            self._saveInputs(inputs)
        if changed or force:
            self._updateAsset(self._buildAsset())

    
    @abstractmethod
    def _buildAsset(self):
        pass
    
class _Point(_MathVisual, metaclass=ABCMeta):
    """
    Abstract base class for all point classes.
    
    :param Asset asset: A valid ggame Asset object
    :param tuple(float,float) pos: The position (physical or logical)
    """

    _posinputsdef = ['pos']
    _nonposinputsdef = []

    def __init__(self, asset, *args, **kwargs):
        super().__init__(asset, *args, **kwargs)
        self._touchAsset()
        self.center = (0.5, 0.5)

    def __call__(self):
        return self._posinputs.pos()

    def step(self):
        """
        Perform periodic processing.
        """
        print("_Point step")
        self._touchAsset()

    def physicalPointTouching(self, ppos):
        """
        Determine if a physical point is considered to be touching this point.
        
        :param tuple(int,int) ppos: Physical screen coordinates.
        :rtype: boolean
        :returns: True if touching, False otherwise.
        """
        return MathApp.distance(ppos, self._pposinputs.pos) < self._sstdinputs.size
        
    def translate(self, pdisp):
        """ 
        Perform necessary processing in response to being moved by the mouse/UI.
        
        :param tuple(int,int) pdisp: Translation vector (x,y) in physical screen
            units.
        :returns: None
        """
        ldisp = MathApp.translatePhysicalToLogical(pdisp)
        pos = self._posinputs.pos()
        self._posinputs = self._posinputs._replace(pos=self.Eval((pos[0] + ldisp[0], pos[1] + ldisp[1])))
        self._touchAsset()
        
    def distanceTo(self, otherpoint):
        """
        Compute the distance to another :class:`_Point` object.
        
        :param _Point otherpoint: A reference to the other :class:`_Point`
        :rtype: float
        :returns: The distance (in logical units) to the other point
        """
        try:
            pos = self._posinputs.pos
            opos = otherpoint._posinputs.pos
            return MathApp.distance(pos, opos())
        except AttributeError:
            return otherpoint  # presumably a scalar - use this distance

class ImagePoint(_Point):
    """
    :class:`~ggame.point.Point` object that uses an image as its on-screen
        representation.
    
    :param str url: Location of an image file (png, jpg)

    :param \*args:
        See below
    :param \**kwargs:
        See below

    :Required Arguments:
        * **pos** (*tuple(float,float)*) Position in physical or logical units.
    

    :Optional Keyword Arguments:
        * **positioning** (*str*) One of 'logical' (default) or 'physical'
        * **frame** (*Frame*) The sub-frame location of image within the image file
        * **qty** (*int*) The number of sub-frames, when used as a sprite sheet
        * **direction** (*str*) One of 'horizontal' (default) or 'vertical'
        * **margin** (*int*) Pixels between sub-frames if sprite sheet
    """


    def __init__(self, url, *args, **kwargs):
        frame = kwargs.get('frame', None)
        qty = kwargs.get('qty', 1)
        direction = kwargs.get('direction', 'horizontal')
        margin = kwargs.get('margin', 0)
        self._imageasset = ImageAsset(url, frame, qty, direction, margin)
        super().__init__(self._imageasset, *args, **kwargs)


    def _buildAsset(self):
        return self._imageasset

    def physicalPointTouching(self, ppos):
        """
        Determine if a physical point is considered to be touching point's 
        image.
        
        :param tuple(int,int) ppos: Physical screen coordinates.
        :rtype: boolean
        :returns: True if touching, False otherwise.
        """
        self._setExtents()  # ensure xmin, xmax are correct
        x, y = ppos
        return x >= self.xmin and x < self.xmax and y >= self.ymin and y <= self.ymax


class Rocket(ImagePoint):
    """
    Rocket is a class for simulating the motion of a projectile through space, 
    acted upon by arbitrary forces (thrust) and by gravitaitonal 
    attraction to a single planetary object.
    """

    def __init__(self, planet, **kwargs):
        """
        Initialize the Rocket object. 
        
        Example:
        
            rocket1 = Rocket(earth, altitude=400000, velocity=7670, timezoom=2)
       
        Required parameters:
        
        * **planet**:  Reference to a `Planet` object.
        
        Optional keyword parameters are supported:
        
        * **bitmap**:  url of a suitable bitmap image for the rocket (png recommended)
          default is `rocket.png`
        * **bitmapscale**:  scale factor for bitmap. Default is 0.1
        * **velocity**:  initial rocket speed. default is zero.
        * **directiond**:  initial rocket direction in degrees. Default is zero.
        * **direction**:  initial rocket direction in radians. Default is zero.
        * **tanomalyd**:  initial rocket true anomaly in degrees. Default is 90.
        * **tanomaly**:  initial rocket true anomaly in radians. Default is pi/2.
        * **altitude**:  initial rocket altitude in meters. Default is zero.
        * **showstatus**:  boolean displays flight parameters on screen. Default
          is True.
        * **statuspos**:  tuple with x,y coordinates of flight parameters. 
          Default is upper left.
        * **statuslist**: list of status names to include in flight parameters. 
          Default is all, consisting of: "velocity", "acceleration", "course",
          "altitude", "thrust", "mass", "trueanomaly", "scale", "timezoom",
          "shiptime"
        * **leftkey**: a `ggame` key identifier that will serve as the 
          "rotate left" key while controlling the ship. Default is 'left arrow'.
        * **rightkey**: a `ggame` key identifier that will serve as the 
          "rotate right" key while controlling the ship. Default is 'right arrow'.
        
        Following parameters may be set as a constant value, or pass in the
        name of a function that will return the value dynamically or the
        name of a `ggmath` UI control that will return the value.
        
        * **timezoom**  scale factor for time zoom. Factor = 10^timezoom
        * **heading**  direction to point the rocket in (must be radians)
        * **mass**  mass of the rocket (must be kg)
        * **thrust**  thrust of the rocket (must be N)

        Animation related parameters may be ignored if no sprite animation:
        
        * **bitmapframe**  ((x1,y1),(x2,y2)) tuple defines a region in the bitmap
        * **bitmapqty**  number of bitmaps -- used for animation effects
        * **bitmapdir**  "horizontal" or "vertical" use with animation effects
        * **bitmapmargin**  pixels between successive animation frames
        * **tickrate**  frequency of spacecraft dynamics calculations (Hz)
        
        """
        self._xy = (1000000,1000000)
        self.planet = planet
        self.bmurl = kwargs.get('bitmap', 'ggimages/rocket.png') # default rocket png
        self.bitmapframe = kwargs.get('bitmapframe', None) #
        self.bitmapqty = kwargs.get('bitmapqty', 1) # Number of images in bitmap
        self.bitmapdir = kwargs.get('bitmapdir', 'horizontal') # animation orientation
        self.bitmapmargin = kwargs.get('bitmapmargin', 0) # bitmap spacing
        self.tickrate = kwargs.get('tickrate', 30) # dynamics calcs per sec
        # status display
        statusfuncs = [ self.velocityText,
                        self.accelerationText,
                        self.courseDegreesText,
                        self.altitudeText,
                        self.thrustText,
                        self.massText,
                        self.trueAnomalyDegreesText,
                        self.scaleText,
                        self.timeZoomText,
                        self.shipTimeText]
        statuslist = [  "velocity",
                        "acceleration",
                        "course",
                        "altitude",
                        "thrust",
                        "mass",
                        "trueanomaly",
                        "scale",
                        "timezoom",
                        "shiptime"]
        
        self.showstatus = kwargs.get('showstatus', True) # show stats
        self.statuspos = kwargs.get('statuspos', [10,10])  # position of stats
        self.statusselect = kwargs.get('statuslist', statuslist)
        self.localheading = 0
        # dynamic parameters
        self.timezoom = self.Eval(kwargs.get('timezoom', self.gettimezoom)) # 1,2,3 faster, -1, slower
        self.heading = self.Eval(kwargs.get('heading', self.getheading)) # must be radians
        self.mass = self.Eval(kwargs.get('mass', self.getmass)) # kg
        self.thrust = self.Eval(kwargs.get('thrust', self.getthrust)) # N
        # end dynamic 
        super().__init__(self.bmurl,
            self._getposition, 
            frame = self.bitmapframe, 
            qty = self.bitmapqty, 
            direction = self.bitmapdir,
            margin = self.bitmapmargin)
        self.scale = kwargs.get('bitmapscale', 0.1) # small
        initvel = kwargs.get('velocity', 0) # initial velocity
        initdird = kwargs.get('directiond', 0) # initial direction, degrees
        initdir = kwargs.get('direction', radians(initdird))
        tanomaly = kwargs.get('tanomaly', pi/2) # position angle
        tanomaly = radians(kwargs.get('tanomalyd', degrees(tanomaly))) 
        altitude = kwargs.get('altitude', 0) #
        r = altitude + self.planet.radius
        self._xy = (r*cos(tanomaly), r*sin(tanomaly))
        # default heading control if none provided by user
        leftkey = kwargs.get('leftkey', 'left arrow')
        rightkey = kwargs.get('rightkey', 'right arrow')
        if self.heading == self.getheading:
            Planet.listenKeyEvent('keydown', leftkey, self.turn)
            Planet.listenKeyEvent('keydown', rightkey, self.turn)
        self.timer = Timer()
        self.shiptime = 0  # track time on shipboard
        self.timer.callEvery(1/self.tickrate, self.dynamics)
        self.lasttime = self.timer.time
        self.V = [initvel * cos(initdir), initvel * sin(initdir)]
        self.A = [0,0]
        # set up status display
        if self.showstatus:
            self.addStatusReport(statuslist, statusfuncs, self.statusselect)

    # override or define externally!
    def getthrust(self):
        """
        User override function for dynamically determining thrust force.
        """
        return 0

    # override or define externally!
    def getmass(self):
        """
        User override function for dynamically determining rocket mass.
        """
        return 1

    # override or define externally!
    def getheading(self):
        """
        User override function for dynamically determining the heading.
        """
        return self.localheading
        
    # override or define externally!
    def gettimezoom(self):
        """
        User override function for dynamically determining the timezoom.
        """
        return 0

    # add a status reporting function to status display
    def addStatusReport(self, statuslist, statusfuncs, statusselect):
        """
        Accept list of all status names, all status text functions, and
        the list of status names that have been selected for display.
        """
        statusdict = {n:f for n, f in zip(statuslist, statusfuncs)}
        for name in statusselect:
            if name in statusdict:
                Label(self.statuspos[:], statusdict[name], size=15, positioning='physical', width=250)
                self.statuspos[1] += 25

    # functions available for reporting flight parameters to UI
    def velocityText(self):
        """
        Report the velocity in m/s as a text string.
        """
        return "Velocity:     {0:8.1f} m/s".format(self.velocity)
        
    def accelerationText(self):
        """
        Report the acceleration in m/s as a text string.
        """
        return "Acceleration: {0:8.1f} m/s²".format(self.acceleration)
        
    def courseDegreesText(self):
        """
        Report the heading in degrees (zero to the right) as a text string.
        """
        return "Course:       {0:8.1f}°".format(degrees(atan2(self.V[1], self.V[0])))

    def thrustText(self):
        """
        Report the thrust level in Newtons as a text string.
        """
        return "Thrust:       {0:8.1f} N".format(self.thrust())
        
    def massText(self):
        """
        Report the spacecraft mass in kilograms as a text string.
        """
        return "Mass:         {0:8.1f} kg".format(self.mass())
        
    def trueAnomalyDegreesText(self):
        """
        Report the true anomaly in degrees as a text string.
        """
        return "True Anomaly: {0:8.1f}°".format(self.tanomalyd)
        
    def trueAnomalyRadiansText(self):
        """
        Report the true anomaly in radians as a text string.
        """
        return "True Anomaly: {0:8.4f}".format(self.tanomaly)
        
    def altitudeText(self):
        """
        Report the altitude in meters as a text string.
        """
        return "Altitude:     {0:8.1f} m".format(self.altitude)
        
    def radiusText(self):
        """
        Report the radius (distance to planet center) in meters as a text string.
        """
        return "Radius:       {0:8.1f} m".format(self.r)
        
    def scaleText(self):
        """
        Report the view scale (pixels/meter) as a text string.
        """
        return "View Scale:   {0:8.6f} px/m".format(self.planet._scale)
    
    def timeZoomText(self):
        """
        Report the time acceleration as a text string.
        """
        return "Time Zoom:    {0:8.1f}".format(float(self.timezoom()))
        
    def shipTimeText(self):
        """
        Report the elapsed time as a text string.
        """
        return "Elapsed Time: {0:8.1f} s".format(float(self.shiptime))
    


            
    def dynamics(self, timer):
        """
        Perform one iteration of the simulation using runge-kutta 4th order method.
        """
        # set time duration equal to time since last execution
        tick = 10**self.timezoom()*(timer.time - self.lasttime)
        self.shiptime = self.shiptime + tick
        self.lasttime = timer.time
        # 4th order runge-kutta method (https://sites.temple.edu/math5061/files/2016/12/final_project.pdf)
        # and http://spiff.rit.edu/richmond/nbody/OrbitRungeKutta4.pdf  (succinct, but with a typo)
        self.A = k1v = self.ar(self._xy)
        k1r = self.V
        k2v = self.ar(self.vadd(self._xy, self.vmul(tick/2, k1r)))
        k2r = self.vadd(self.V, self.vmul(tick/2, k1v))
        k3v = self.ar(self.vadd(self._xy, self.vmul(tick/2, k2r)))
        k3r = self.vadd(self.V, self.vmul(tick/2, k2v))
        k4v = self.ar(self.vadd(self._xy, self.vmul(tick, k3r)))
        k4r = self.vadd(self.V, self.vmul(tick, k3v))
        self.V = [self.V[i] + tick/6*(k1v[i] + 2*k2v[i] + 2*k3v[i] + k4v[i]) for i in (0,1)]
        self._xy = [self._xy[i] + tick/6*(k1r[i] + 2*k2r[i] + 2*k3r[i] + k4r[i]) for i in (0,1)]
        if self.altitude < 0:
            self.V = [0,0]
            self.A = [0,0]
            self.altitude = 0

    # generic force as a function of position
    def fr(self, pos):
        """
        Compute the net force vector on the rocket, as a function of the 
        position vector.
        """
        self.rotation = self.heading()
        t = self.thrust()
        G = 6.674E-11
        r = Planet.distance((0,0), pos)
        uvec = (-pos[0]/r, -pos[1]/r)
        fg = G*self.mass()*self.planet.mass/r**2
        F = [x*fg for x in uvec]
        return [F[0] + t*cos(self.rotation), F[1] + t*sin(self.rotation)]

    # geric acceleration as a function of position
    def ar(self, pos):
        """
        Compute the acceleration vector of the rocket, as a function of the 
        position vector.
        """
        m = self.mass()
        F = self.fr(pos)
        return [F[i]/m for i in (0,1)]
        
    def vadd(self, v1, v2):
        """
        Vector add utility.
        """
        return [v1[i]+v2[i] for i in (0,1)]
    
    def vmul(self, s, v):
        """
        Scalar vector multiplication utility.
        """
        return [s*v[i] for i in (0,1)]
        
    def vmag(self, v):
        """
        Vector magnitude function.
        """
        return sqrt(v[0]**2 + v[1]**2)
    
    def fgrav(self):
        """
        Vector force due to gravity, at current position.
        """
        G = 6.674E-11
        r = self.r
        uvec = (-self._xy[0]/r, -self._xy[1]/r)
        F = G*self.mass()*self.planet.mass/r**2
        return [x*F for x in uvec]
    
    def turn(self, event):
        """
        Respond to left/right turning key events.
        """
        increment = pi/50 * (1 if event.key == "left arrow" else -1)
        self.localheading += increment
            
    def _getposition(self):
        return self._xy
    
    @property
    def xyposition(self):
        return self._xy
        
    @xyposition.setter
    def xyposition(self, pos):
        self._xy = pos
        #self._touchAsset()

    @property
    def tanomalyd(self):
        return degrees(self.tanomaly)
        
    @tanomalyd.setter
    def tanomalyd(self, angle):
        self.tanomaly = radians(angle)

    @property
    def altitude(self):
        alt = Planet.distance(self._xy, (0,0)) - self.planet.radius
        return alt
        
    @altitude.setter
    def altitude(self, alt):
        r = alt + self.planet.radius
        self._xy = (r*cos(self.tanomaly), r*sin(self.tanomaly))

    @property
    def velocity(self):
        return self.vmag(self.V)
    
    @property
    def acceleration(self):
        return self.vmag(self.A)
        
    @property
    def tanomaly(self):
        #pos = self._pos()
        return atan2(self._xy[1],self._xy[0])
        
    @tanomaly.setter
    def tanomaly(self, angle):
        r = self.r
        self._xy = (r*cos(angle), r*sin(angle))
        self._touchAsset()
            
    @property
    def r(self):
        return self.altitude + self.planet.radius

        
        
class MathApp(App):
    """
    MathApp is a subclass of the ggame :class:`~ggame.app.App` class. It 
    incorporates the following extensions:
    
    * Support for zooming the display using the mouse wheel
    * Support for click-dragging the display using the mouse button
    * Automatic execution of step functions in all objects and sprites
        sub-classed from :class:`_MathDynamic`.
        
    :param float scale: Optional parameter sets the initial scale of the
        display in units of pixels per logical unit. The default is 200.
        
    :returns: MathApp instance
    """
    
    _scale = 200   # pixels per unit
    _xcenter = 0    # center of screen in units
    _ycenter = 0    
    _mathVisualList = [] #
    _mathDynamicList = []
    _mathMovableList = []
    _mathSelectableList = []
    _mathStrokableList = []
    _viewNotificationList = []
    time = time()
    
    def __init__(self, scale=_scale):
        super().__init__()
        MathApp.width = self.width
        MathApp.height = self.height
        MathApp._scale = scale   # pixels per unit
        # register event callbacks
        self.listenMouseEvent("click", self._handleMouseClick)
        self.listenMouseEvent("mousedown", self._handleMouseDown)
        self.listenMouseEvent("mouseup", self._handleMouseUp)
        self.listenMouseEvent("mousemove", self._handleMouseMove)
        self.listenMouseEvent("wheel", self._handleMouseWheel)
        self.mouseDown = False
        self.mouseCapturedObject = None
        self.mouseStrokedObject = None
        self.mouseDownObject = None
        self.mouseX = self.mouseY = None
        self._touchAllVisuals()
        self.selectedObj = None
        MathApp.time = time()

    def step(self):
        """
        The step method overrides :func:`~ggame.app.App.step` in the 
        :class:`~ggame.app.App` class, executing step functions in all
        objects subclassed from :class:`_MathDynamic`.
        """
        MathApp.time = time()
        for spr in self._mathDynamicList:
            spr.step()

    def _touchAllVisuals(self):
        # touch all visual object assets to use scaling
        for obj in self._mathVisualList:
            obj._touchAsset(True)


    @classmethod
    def logicalToPhysical(cls, lp):
        """
        Transform screen coordinates from logical to physical space. Output
        depends on the current 'zoom' and 'pan' of the screen.
        
        :param tuple(float,float) lp: Logical screen coordinates (x, y)
        
        :rtype: tuple(float,float)
        
        :returns: Physical screen coordinates (x, y)
        """
        
        xxform = lambda xvalue, xscale, xcenter, physwidth: int((xvalue-xcenter)*xscale + physwidth/2)
        yxform = lambda yvalue, yscale, ycenter, physheight: int(physheight/2 - (yvalue-ycenter)*yscale)

        try:
            return (xxform(lp[0], cls._scale, cls._xcenter, cls._win.width),
                yxform(lp[1], cls._scale, cls._ycenter, cls._win.height))
        except AttributeError:
            return lp
            
    @classmethod
    def physicalToLogical(cls, pp):
        """
        Transform screen coordinates from physical to logical space. Output
        depends on the current 'zoom' and 'pan' of the screen.
        
        :param tuple(float,float) lp: Physical screen coordinates (x, y)
        
        :rtype: tuple(float,float)
        
        :returns: Logical screen coordinates (x, y)
        """
        
        xxform = lambda xvalue, xscale, xcenter, physwidth: (xvalue - physwidth/2)/xscale + xcenter
        yxform = lambda yvalue, yscale, ycenter, physheight: (physheight/2 - yvalue)/yscale + ycenter

        try:
            return (xxform(pp[0], cls._scale, cls._xcenter, cls._win.width),
                yxform(pp[1], cls._scale, cls._ycenter, cls._win.height))
        except AttributeError:
            return pp
            
    @classmethod
    def translateLogicalToPhysical(cls, pp):
        """
        Transform screen translation from logical to physical space. Output
        only depends on the current 'zoom' of the screen.
        
        :param tuple(float,float) lp: Logical screen translation pair 
            (delta x, delta y)
        
        :rtype: tuple(float,float)
        
        :returns: Physical screen translation ordered pair (delta x, delta y)
        """
        
        xxform = lambda xvalue, xscale: xvalue*xscale
        yxform = lambda yvalue, yscale: -yvalue*yscale

        try:
            return (xxform(pp[0], cls._scale), yxform(pp[1], cls._scale))
        except AttributeError:
            return pp

    @classmethod
    def translatePhysicalToLogical(cls, pp):
        """
        Transform screen translation from physical to logical space. Output
        only depends on the current 'zoom' of the screen.
        
        :param tuple(float,float) lp: Physical screen translation pair 
            (delta x, delta y)
        
        :rtype: tuple(float,float)
        
        :returns: Logical screen translation ordered pair (delta x, delta y)
        """
        
        xxform = lambda xvalue, xscale: xvalue/xscale
        yxform = lambda yvalue, yscale: -yvalue/yscale

        try:
            return (xxform(pp[0], cls._scale), yxform(pp[1], cls._scale))
        except AttributeError:
            return pp

    def _handleMouseClick(self, event):
        found = False
        for obj in self._mathSelectableList:
            if obj.physicalPointTouching((event.x, event.y)):
                found = True
                if not obj.selected: 
                    obj.select()
                    self.selectedObj = obj
        if not found and self.selectedObj:
            self.selectedObj.unselect()
            self.selectedObj = None

    def _handleMouseDown(self, event):
        self.mouseDown = True
        self.mouseCapturedObject = None
        self.mouseStrokedObject = None
        for obj in self._mathSelectableList:
            if obj.physicalPointTouching((event.x, event.y)):
                obj.mousedown()
                self.mouseDownObject = obj
                break
        for obj in self._mathMovableList:
            if obj.physicalPointTouching((event.x, event.y)) and not (obj.strokable and obj.canstroke((event.x,event.y))):
                self.mouseCapturedObject = obj
                break
        if not self.mouseCapturedObject:
            for obj in self._mathStrokableList:
                if obj.canstroke((event.x, event.y)):
                    self.mouseStrokedObject = obj
                    break

    def _handleMouseUp(self, event):
        if self.mouseDownObject:
            self.mouseDownObject.mouseup()
            self.mouseDownObject = None
        self.mouseDown = False
        self.mouseCapturedObject = None
        self.mouseStrokedObject = None

    def _handleMouseMove(self, event):
        if not self.mouseX:
            self.mouseX = event.x
            self.mouseY = event.y
        dx = event.x - self.mouseX
        dy = event.y - self.mouseY
        self.mouseX = event.x
        self.mouseY = event.y
        if self.mouseDown:
            if self.mouseCapturedObject:
                self.mouseCapturedObject.translate((dx, dy))
            elif self.mouseStrokedObject:
                self.mouseStrokedObject.stroke((self.mouseX,self.mouseY), (dx,dy))
            else:
                lmove = self.translatePhysicalToLogical((dx, dy))
                MathApp._xcenter -= lmove[0]
                MathApp._ycenter -= lmove[1]
                self._touchAllVisuals()
                self._viewNotify("translate")
    
    def _handleMouseWheel(self, event):
        zoomfactor = event.wheelDelta/100
        zoomfactor = 1+zoomfactor if zoomfactor > 0 else 1+zoomfactor
        if zoomfactor > 1.2:
            zoomfactor = 1.2
        elif zoomfactor < 0.8:
            zoomfactor = 0.8
        MathApp._scale *= zoomfactor
        self._touchAllVisuals()
        self._viewNotify("zoom")
        
    @property
    def viewPosition(self):
        """
        Attribute is used to get or set the current logical coordinates 
        at the center of the screen as a tuple of floats (x,y).
        """
        return (MathApp._xcenter, MathApp._ycenter)
        
    @viewPosition.setter
    def viewPosition(self, pos):
        MathApp._xcenter, MathApp._ycenter = pos
        self._touchAllVisuals()
        self._viewNotify("translate")
        
    @classmethod   
    def addViewNotification(cls, handler):
        """
        Register a function or method to be called in the event the view
        position or zoom changes.
        
        :param function handler: The function or method to be called
        :returns: Nothing
        """
        cls._viewNotificationList.append(handler)
        
    @classmethod   
    def removeViewNotification(cls, handler):
        """
        Remove a function or method from the list of functions to be called
        in the event of a view position or zoom change.
        
        :param function handler: The function or method to be removed
        :returns: Nothing
        """
        cls._viewNotificationList.remove(handler)
    
    def _viewNotify(self, viewchange):
        for handler in self._viewNotificationList:
            handler(viewchange = viewchange, scale = self._scale, center = (self._xcenter, self._ycenter))
        
     
    @classmethod   
    def distance(cls, pos1, pos2):
        """
        Utility for calculating the distance between any two points.
        
        :param tuple(float,float) pos1: The first point
        :param tuple(float,float) pos2: The second point
        :rtype: float
        :returns: The distance between the two points (using Pythagoras)
        """
        return sqrt((pos2[0]-pos1[0])**2 + (pos2[1]-pos1[1])**2)
        
    @property
    def scale(self):
        """
        Attribute reports the current view scale (pixels per logical unit).
        """
        return self._scale
        
    @property
    def width(self):
        """
        Attribute reports the physical screen width (pixels).
        """
        return App._win.width

    @width.setter
    def width(self, value):
        pass

            
    @classmethod
    def _addVisual(cls, obj):
        """ FIX ME """
        if isinstance(obj, _MathVisual):
            cls._mathVisualList.append(obj)
            
    @classmethod
    def _removeVisual(cls, obj):
        if isinstance(obj, _MathVisual) and obj in cls._mathVisualList:
            cls._mathVisualList.remove(obj)

    @classmethod
    def _addDynamic(cls, obj):
        print("MathApp _addDynamic", obj)
        if isinstance(obj, _MathDynamic) and not obj in cls._mathDynamicList:
            print("Adding...")
            cls._mathDynamicList.append(obj)
            
    @classmethod
    def _removeDynamic(cls, obj):
        if isinstance(obj, _MathDynamic) and obj in cls._mathDynamicList:
            cls._mathDynamicList.remove(obj)

    @classmethod
    def _addMovable(cls, obj):
        if isinstance(obj, _MathVisual) and not obj in cls._mathMovableList:
            cls._mathMovableList.append(obj)
            
    @classmethod
    def _removeMovable(cls, obj):
        if isinstance(obj, _MathVisual) and obj in cls._mathMovableList:
            cls._mathMovableList.remove(obj)

    @classmethod
    def _addSelectable(cls, obj):
        if isinstance(obj, _MathVisual) and not obj in cls._mathSelectableList:
            cls._mathSelectableList.append(obj)
            
    @classmethod
    def _removeSelectable(cls, obj):
       if isinstance(obj, _MathVisual)  and obj in cls._mathSelectableList:
            cls._mathSelectableList.remove(obj)

    @classmethod
    def _addStrokable(cls, obj):
        if isinstance(obj, _MathVisual) and not obj in cls._mathStrokableList:
            cls._mathStrokableList.append(obj)
            
    @classmethod
    def _removeStrokable(cls, obj):
        if isinstance(obj, _MathVisual) and obj in cls._mathStrokableList:
            cls._mathStrokableList.remove(obj)

    @classmethod
    def _destroy(cls, *args):
        """
        This will clean up any class level storage.
        """ 
        App._destroy(*args)  # hit the App class first
        MathApp._mathVisualList = [] 
        MathApp._mathDynamicList = []
        MathApp._mathMovableList = []
        MathApp._mathSelectableList = []
        MathApp._mathStrokableList = []
        MathApp._viewNotificationList = []
     


class Planet(MathApp):
    
    def __init__(self, **kwargs):
        """
        Initialize the Planet object. 

        Optional keyword parameters are supported:
        
        * **viewscale**  pixels per meter in graphics display. Default is 10.
        * **radius**  radius of the planet in meters. Default is Earth radius.
        * **planetmass** mass of the planet in kg. Default is Earth mass.
        * **color** color of the planet. Default is greenish (0x008040).
        * **viewalt** altitude of initial viewpoint in meters. Default is rocket 
          altitude.
        * **viewanom** true anomaly (angle) of initial viewpoint in radians. 
          Default is the rocket anomaly.
        * **viewanomd** true anomaly (angle) of initial viewpoing in degrees.
          Default is the rocket anomaly.
        
        """
        self.scale = kwargs.get('viewscale', 10)  # 10 pixels per meter default
        self.radius = kwargs.get('radius', 6.371E6) # Earth - meters
        self.mass = kwargs.get('planetmass', 5.9722E24) # Earth - kg
        self.color = kwargs.get('color', 0x008040)  # greenish
        self.kwargs = kwargs # save it for later..
        super().__init__(self.scale)

    def run(self, rocket=None):
        """
        Execute the Planet (and Rocket) simulation.

        Optional parameters:
        
        * **rocket** Reference to a Rocket object - sets the initial view
        """
        if rocket:
            viewalt = rocket.altitude
            viewanom = rocket.tanomaly
        else:
            viewalt = 0
            viewanom = pi/2
        self.viewaltitude = self.kwargs.get('viewalt', viewalt) # how high to look
        self.viewanomaly = self.kwargs.get('viewanom', viewanom)  # where to look
        self.viewanomalyd = self.kwargs.get('viewanomd', degrees(self.viewanomaly))
        self.planetcircle = Circle(
            (0,0), 
            self.radius, 
            style = LineStyle(1, Color(self.color,1)), 
            color = Color(self.color,0.5))
        r = self.radius + self.viewaltitude
        self.viewPosition = (r*cos(self.viewanomaly), r*sin(self.viewanomaly))
        super().run()

# test code here
if __name__ == "__main__":
    
    earth = Planet(viewscale=0.00005)
    earth.run()

    rocket1 = Rocket(earth, altitude=400000, velocity=7670, timezoom=2)
    rocket2 = Rocket(earth, altitude=440000, velocity=7670, timezoom=2, statuspos=[300,10])

