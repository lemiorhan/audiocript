"""The menu bar: two status items, their menus, and the run loop that owns the main
thread.

Nothing here decides what the user sees. Which glyph an icon shows and what a menu
contains comes from dictation's menu model; this file turns MenuItem lists into
NSMenu objects, dispatches the actions they carry, and keeps the main thread alive.
A branch on power or state appearing in here belongs on the other side of that line,
where it can be tested — there is no headless AppKit.

Every AppKit behaviour this module relies on was measured on this machine before it
was written, because none of it is checkable from a test that does not click:

- A non-bundled Python process can own two NSStatusItems, both visible, with emoji
  titles, under NSApplicationActivationPolicyAccessory (no Dock icon).
- A PyObjC subclass instance carries a plain Python attribute, which is how the click
  target reaches its handler.
- NSApp.sendAction_to_from_ delivers a menu item's action to that target with the
  item's tag intact.
- setAppearsDisabled_ and setToolTip_ both round-trip. Whether they *look* right is
  the one thing left for a human to check.
- app.stop_(None) does NOT end the run loop on its own; paired with a posted
  application-defined event it does, and app.run() returns to its caller — measured
  at 804 ms from a signal handler, with the caller's `finally` reached. NSApp's
  terminate_ ends the loop too, but by ending the process: the pid file would be left
  behind and the next start refused. So nothing here calls it.
"""
import subprocess

import AppKit
import Foundation

import audiocript as A
import dictation

# How often the main thread wakes up, and why it must.
#
# Python runs a signal handler only between bytecodes on the main thread, and
# NSApp.run() spends its life inside Objective-C. Measured: with no timer scheduled at
# all, a SIGUSR1 handler ran 4996 ms after the signal arrived; with this timer, 197 ms.
# `dictate.py --toggle`, `--stop` and Ctrl-C all depend on it, so the timer is
# load-bearing rather than a convenience — and the interval is the latency bound on
# all three.
#
# It is also what refreshes the icons and the menus; see MenuBar.refresh for why that
# is a poll rather than a report.
HEARTBEAT_SECONDS = 0.2

# The selector every menu item's click arrives on.
CLICK_ACTION = "menuItemClicked:"


class _Clicks(AppKit.NSObject):
    """The Objective-C target every menu item points at.

    AppKit needs a real target, so this is an NSObject subclass; `handler` is assigned
    from Python after alloc/init, which a PyObjC subclass supports. The handler is
    given the clicked item's tag, because a tag is the only payload AppKit carries
    from an item to its action for free.
    """

    def menuItemClicked_(self, sender):
        self.handler(sender.tag())


def render(nsmenu, items, target):
    """Replace `nsmenu`'s contents with `items`, and return the action table.

    The table is a list of (action, payload) indexed by the tag each row was given. It
    is rebuilt with the menu, so a tag cannot outlive the row it belonged to.

    autoenablesItems is switched off. Left on — the default — AppKit decides for
    itself whether a row with a target and an action is clickable, and it enables
    every row whose target responds to the selector. That would quietly undo the one
    thing the menu model is most careful about: "Daemon'ı durdur" greyed out while a
    dictation is running.
    """
    nsmenu.setAutoenablesItems_(False)
    nsmenu.removeAllItems()
    table = []
    for item in items:
        if item.separator:
            nsmenu.addItem_(AppKit.NSMenuItem.separatorItem())
            continue
        row = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            item.title, CLICK_ACTION if item.action else None, "")
        # A row with no action is a label — the stage the daemon is in, or a note that
        # there is nothing to list — and is never clickable whatever the model said.
        row.setEnabled_(bool(item.action) and item.enabled)
        if item.action:
            row.setTarget_(target)
            row.setTag_(len(table))
            table.append((item.action, item.payload))
        if item.payload:
            # The whole dictation, for hovering: the title is cut to 250 characters,
            # and macOS narrows it further at its own discretion.
            row.setToolTip_(item.payload)
        nsmenu.addItem_(row)
    return table


def fingerprint(power, state, history_path):
    """What the icons and menus are drawn from, reduced to something comparable.

    The log contributes its size and modification time rather than its contents: the
    menus have to notice a new dictation, and reading the file on every heartbeat
    would be five reads a second to catch a change that happens once a minute at most.
    """
    try:
        stat = history_path.stat()
        log = (stat.st_size, stat.st_mtime_ns)
    except OSError:
        log = None
    return power, state, log


def stop_the_loop():
    """Make the run loop return to whoever called app.run().

    Both halves are needed. stop_() alone was measured not to end the loop — it is a
    no-op until the loop next handles an event, and an idle menu bar may not see one
    for minutes — so an application-defined event is posted to wake it. Returning
    rather than terminating is the point: the caller's `finally` is where the pid file
    is removed and a dictation in flight is drained.
    """
    app = AppKit.NSApplication.sharedApplication()
    app.stop_(None)
    event = AppKit.NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
        AppKit.NSEventTypeApplicationDefined, Foundation.NSZeroPoint, 0, 0, 0, None,
        0, 0, 0)
    app.postEvent_atStart_(event, True)


class MenuBar:
    """Both status items, and the refresh that keeps them true.

    The icons and menus are derived from the daemon's own two attributes on every
    heartbeat, not pushed by a status sink. That is deliberate: the sink's moments are
    content reports, not state transitions, and an icon drawn from them goes stale
    three different ways —

    - `done` and `failed` are sent by the worker before _work's `finally` returns the
      daemon to IDLE, so an icon drawn on `done` shows PROCESSING;
    - `failed` is also how a refusal is reported ("the daemon is not running", "a
      dictation is in progress"), where nothing has ended and the icon must not move;
    - _begin_processing's failure path returns to IDLE having already sent
      `recording` and never sending `processing`, so a 🔴 would stay on screen with
      nothing running behind it.

    Closing those by hand means every path back to IDLE has to remember to report —
    four call sites to keep in step, and the fifth one added later is silent. A poll
    over (power, state) cannot go stale, and the timer it rides on has to exist anyway
    for the signal handlers.
    """

    def __init__(self, daemon, history, clipboard, bar=None):
        self._daemon = daemon
        self._history = history
        self._clipboard = clipboard
        bar = bar or AppKit.NSStatusBar.systemStatusBar()
        # Creation order is position, and position decides which of the two a menu
        # bar manager swallows. NSStatusBar puts the first item created rightmost,
        # nearest the clock, and every later one to its left — measured on this
        # machine, where the first item's window sat at x=1054 on a 1920-wide screen
        # and the second at x=-4577, which is where Ice had moved it out of sight.
        #
        # So the dictation icon is created first. It is the one used constantly; the
        # power icon is clicked twice a day, and it is the one that can afford to be
        # the one hidden.
        self._dictation_item = bar.statusItemWithLength_(
            AppKit.NSVariableStatusItemLength)
        self._power_item = bar.statusItemWithLength_(
            AppKit.NSVariableStatusItemLength)
        self._power_menu = AppKit.NSMenu.alloc().init()
        self._dictation_menu = AppKit.NSMenu.alloc().init()
        self._power_item.setMenu_(self._power_menu)
        self._dictation_item.setMenu_(self._dictation_menu)
        # One target per menu, so a tag is unambiguous: the two tables are rebuilt
        # independently and a tag means nothing without knowing which one it indexes.
        self._power_table, self._dictation_table = [], []
        self._power_target = _Clicks.alloc().init()
        self._power_target.handler = lambda tag: self._invoke(self._power_table, tag)
        self._dictation_target = _Clicks.alloc().init()
        self._dictation_target.handler = \
            lambda tag: self._invoke(self._dictation_table, tag)
        self._drawn = None
        self.refresh()

    # --------------------------------- drawing ---------------------------------

    def refresh(self):
        """Redraw both icons and both menus if anything they show has changed.

        Called on every heartbeat and after every menu action, always on the main
        thread. The fingerprint comparison is what keeps five wake-ups a second from
        being five redraws and five history reads.
        """
        power, state = self._daemon.power, self._daemon.state
        current = fingerprint(power, state, self._history.path)
        if current == self._drawn:
            return
        self._drawn = current
        self._power_item.button().setTitle_(dictation.power_icon(power))
        title, dimmed = dictation.dictation_icon(power, state)
        button = self._dictation_item.button()
        button.setTitle_(title)
        button.setAppearsDisabled_(dimmed)
        self._power_table = render(self._power_menu,
                                   dictation.power_menu(power, state),
                                   self._power_target)
        self._dictation_table = render(
            self._dictation_menu,
            dictation.dictation_menu(power, state, self._entries()),
            self._dictation_target)

    def _entries(self):
        """The newest dictations, or None when the log could not be read.

        History.recent deliberately does not swallow, so that this can tell the
        difference between "no dictations yet" and "the log is unreadable" — the menu
        saying the first when the second is true is the one lie it must not tell.
        """
        try:
            return self._history.recent()
        except Exception as e:
            A._log_problem("the dictation history could not be read", e)
            return None

    # -------------------------------- the actions --------------------------------

    def _invoke(self, table, tag):
        """Run what a clicked row carried.

        Guarded twice over. An exception here would escape into AppKit's own action
        dispatch, where it is a wedged menu bar rather than a traceback anybody sees;
        and a tag with no row behind it means a menu was clicked as it was being
        rebuilt, which is a thing to log rather than to crash on.
        """
        try:
            action, payload = table[tag]
        except IndexError:
            A._log_problem(f"a menu row's tag ({tag}) outlived its table", None)
            return
        try:
            self._perform(action, payload)
        except Exception as e:
            A._log_problem(f"the menu action {action!r} failed", e)

    def _perform(self, action, payload):
        if action == "power_on":
            self._daemon.enable()
        elif action == "power_off":
            self._daemon.disable()
        elif action == "record_toggle":
            self._daemon.toggle()
        elif action == "copy":
            # No report. The user asked for this by clicking it, and a notification
            # for every click would be noise.
            self._clipboard.copy(payload)
        elif action == "reveal":
            self._reveal()
        elif action == "quit":
            stop_the_loop()
            return                      # nothing left to redraw; the loop is ending
        else:
            A._log_problem(f"unknown menu action {action!r}", None)
            return
        # At once rather than on the next heartbeat: a click that took up to 200 ms to
        # show anything reads as a click that missed.
        self.refresh()

    def _reveal(self):
        """Show the log in Finder.

        Before the first dictation there is no file, so the folder it will appear in
        is the next best thing: `open -R` on a missing path exits non-zero and prints
        a message nobody would see from a menu (measured).
        """
        path = self._history.path
        argv = (["open", "-R", str(path)] if path.exists()
                else ["open", str(path.parent)])
        subprocess.run(argv, check=True, timeout=dictation.NOTIFY_TIMEOUT_SECONDS)


def run(daemon, history, clipboard):
    """Put both icons in the menu bar and run the loop until something stops it.

    Returns when the loop does, so the caller's shutdown runs — see stop_the_loop.
    """
    app = AppKit.NSApplication.sharedApplication()
    # Accessory: no Dock icon, no menu of its own. Verified on this machine.
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
    menu_bar = MenuBar(daemon, history, clipboard)

    def tick(_timer):
        try:
            menu_bar.refresh()
        except Exception as e:
            # The timer's other job — giving the interpreter main-thread ticks so
            # signal handlers run — must survive a redraw that cannot.
            A._log_problem("the menu bar could not be refreshed", e)

    heartbeat = Foundation.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
        HEARTBEAT_SECONDS, True, tick)
    try:
        app.run()
    finally:
        heartbeat.invalidate()
