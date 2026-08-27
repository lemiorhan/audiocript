"""menubar: turning the menu model into NSMenu objects, and back into actions.

More of this is testable than the plan expected. MenuBar takes the status bar as a
parameter, so a fake one keeps the two icons out of the real menu bar; NSMenu and
NSMenuItem need no window; and AppKit's own action dispatch can be driven directly
through NSApp.sendAction_to_from_, which is how every click below is delivered.

What is left for a human: whether a dimmed icon looks dimmed, whether a tooltip
appears on hover, and whether a menu opens where the pointer is. Those are in the
spec's acceptance criteria, not here.

No test here runs the loop. NSApplication.sharedApplication() is created because
sendAction_to_from_ needs an application object, but app.run() is never called.
"""
import pathlib
import threading

import AppKit

from support import A, run, workdir

import dictation
import menubar

AppKit.NSApplication.sharedApplication().setActivationPolicy_(
    AppKit.NSApplicationActivationPolicyAccessory)


class FakeButton:
    def __init__(self):
        self.title, self.dimmed = None, None

    def setTitle_(self, title):
        self.title = title

    def setAppearsDisabled_(self, dimmed):
        self.dimmed = dimmed


class FakeStatusItem:
    def __init__(self):
        self._button, self.menu = FakeButton(), None

    def button(self):
        return self._button

    def setMenu_(self, menu):
        self.menu = menu


class FakeStatusBar:
    """Keeps the test's icons out of the real menu bar."""

    def __init__(self):
        self.items = []

    def statusItemWithLength_(self, _length):
        self.items.append(FakeStatusItem())
        return self.items[-1]


class FakeDaemon:
    def __init__(self, power=dictation.POWER_OFF, state=dictation.IDLE):
        self.power, self.state, self.calls = power, state, []

    def enable(self):
        self.calls.append("enable")
        self.power = dictation.POWER_LOADING

    def disable(self):
        self.calls.append("disable")
        self.power = dictation.POWER_OFF

    def toggle(self):
        self.calls.append("toggle")


class FakeClipboard:
    def __init__(self):
        self.text = None

    def copy(self, text):
        self.text = text


def _bar(home, power=dictation.POWER_OFF, state=dictation.IDLE, history=None):
    """A MenuBar over a fake status bar and a real History in a temporary directory."""
    daemon = FakeDaemon(power, state)
    history = history or dictation.History(pathlib.Path(home) / "dictations.jsonl")
    clipboard = FakeClipboard()
    bar = FakeStatusBar()
    return menubar.MenuBar(daemon, history, clipboard, bar=bar), daemon, history, \
        clipboard, bar


def _click(nsmenu, table, action):
    """Click the row carrying `action`, through AppKit's own action dispatch."""
    tags = [index for index, (carried, _payload) in enumerate(table)
            if carried == action]
    assert tags, f"no row carries {action!r}: {table!r}"
    for index in range(nsmenu.numberOfItems()):
        row = nsmenu.itemAtIndex_(index)
        if row.action() and row.tag() == tags[0]:
            assert AppKit.NSApp().sendAction_to_from_(row.action(), row.target(), row), \
                f"AppKit refused to dispatch {action!r}"
            return
    raise AssertionError(f"no menu row found for {action!r}, tag {tags[0]}")


def _titles(nsmenu):
    return [nsmenu.itemAtIndex_(i).title() if not nsmenu.itemAtIndex_(i).isSeparatorItem()
            else "---" for i in range(nsmenu.numberOfItems())]


def test_render_puts_every_row_in_the_menu():
    items = [dictation.MenuItem("first", "quit"),
             dictation.MenuItem("", separator=True),
             dictation.MenuItem("a label", enabled=False)]
    nsmenu = AppKit.NSMenu.alloc().init()
    target = menubar._Clicks.alloc().init()
    table = menubar.render(nsmenu, items, target)
    assert _titles(nsmenu) == ["first", "---", "a label"], _titles(nsmenu)
    assert table == [("quit", None)], table
    print(f"  {_titles(nsmenu)}")


def test_appkit_cannot_re_enable_a_row_the_model_disabled():
    """The gotcha this whole file exists to catch. With autoenablesItems left on — the
    default — NSMenu decides for itself whether a row with a target and an action is
    clickable, and enables every row whose target responds to the selector. That would
    silently undo "Daemon'ı durdur" being greyed out while a dictation is running,
    which is the one rule the user asked for by name."""
    disabled = [dictation.MenuItem("Daemon'ı durdur", "power_off", enabled=False)]
    nsmenu = AppKit.NSMenu.alloc().init()
    target = menubar._Clicks.alloc().init()
    target.handler = lambda _tag: None
    menubar.render(nsmenu, disabled, target)
    row = nsmenu.itemAtIndex_(0)
    assert not row.isEnabled(), "render did not disable the row at all"
    nsmenu.update()                     # what AppKit does before drawing
    assert not row.isEnabled(), \
        "AppKit re-enabled a row the model disabled; autoenablesItems is not off"
    assert not nsmenu.autoenablesItems(), "autoenablesItems was left on"
    print("  a disabled row survives NSMenu.update()")


def test_a_row_with_no_action_is_never_clickable():
    """A label — the stage the daemon is in, or a note that there is nothing to list —
    must not look live even if a model row arrived with enabled left true."""
    nsmenu = AppKit.NSMenu.alloc().init()
    menubar.render(nsmenu, [dictation.MenuItem("İşleniyor…", None, enabled=True)],
                   menubar._Clicks.alloc().init())
    assert not nsmenu.itemAtIndex_(0).isEnabled(), "a label was left clickable"


def test_a_copy_row_carries_its_whole_text_as_a_tooltip():
    spoken = "Söz " * 200
    nsmenu = AppKit.NSMenu.alloc().init()
    table = menubar.render(
        nsmenu, [dictation.MenuItem(dictation.menu_preview(spoken), "copy",
                                    payload=spoken)],
        menubar._Clicks.alloc().init())
    row = nsmenu.itemAtIndex_(0)
    assert row.toolTip() == spoken, "the tooltip is not the whole dictation"
    assert len(row.title()) < len(spoken), "the title was not cut"
    assert table == [("copy", spoken)], table
    print(f"  title {len(row.title())} chars, tooltip {len(row.toolTip())}")


def test_every_click_reaches_the_daemon():
    """Driven through AppKit's dispatch, not by calling the handler: the wiring — the
    selector name, the target, the tag — is the part that can silently be wrong."""
    with workdir("menubar-clicks") as home:
        bar, daemon, history, clipboard, _ = _bar(home)
        _click(bar._power_menu, bar._power_table, "power_on")
        assert daemon.calls == ["enable"], daemon.calls

        bar._daemon.power, bar._daemon.state = dictation.POWER_ON, dictation.IDLE
        bar.refresh()
        _click(bar._dictation_menu, bar._dictation_table, "record_toggle")
        assert daemon.calls == ["enable", "toggle"], daemon.calls
        _click(bar._power_menu, bar._power_table, "power_off")
        assert daemon.calls == ["enable", "toggle", "disable"], daemon.calls

        history.append("Toplantıyı yarına alalım.", dictation.CORRECTED)
        bar.refresh()
        _click(bar._dictation_menu, bar._dictation_table, "copy")
        assert clipboard.text == "Toplantıyı yarına alalım.", clipboard.text
        print(f"  {daemon.calls}, clipboard={clipboard.text!r}")


def test_quit_stops_the_loop_instead_of_the_process():
    """terminate_ would end the process outright and the pid file would outlive it,
    refusing the next start. Stopping the loop returns from app.run() so
    _run_daemon's finally removes it."""
    stopped = []
    saved = menubar.stop_the_loop
    menubar.stop_the_loop = lambda: stopped.append(True)
    try:
        with workdir("menubar-quit") as home:
            bar, daemon, _h, _c, _b = _bar(home)
            _click(bar._power_menu, bar._power_table, "quit")
    finally:
        menubar.stop_the_loop = saved
    assert stopped == [True], "the quit row did not stop the loop"
    assert daemon.calls == [], f"quit touched the daemon: {daemon.calls!r}"


def test_a_stale_tag_is_logged_rather_than_raised():
    """A menu clicked while it is being rebuilt. An IndexError here escapes into
    AppKit's own dispatch, where it is a wedged menu bar rather than a traceback
    anybody sees."""
    logged = []
    saved = A._log_problem
    A._log_problem = lambda what, exc=None: logged.append(what)
    try:
        with workdir("menubar-stale") as home:
            bar, _d, _h, _c, _b = _bar(home)
            bar._invoke([], 3)
    finally:
        A._log_problem = saved
    assert logged and "tag" in logged[0], f"nothing useful was logged: {logged!r}"
    print(f"  {logged[0]}")


def test_an_action_that_raises_does_not_escape():
    logged = []
    saved = A._log_problem
    A._log_problem = lambda what, exc=None: logged.append(what)
    try:
        with workdir("menubar-raise") as home:
            bar, daemon, _h, _c, _b = _bar(home)
            def angry():
                raise OSError("the microphone is on fire")
            daemon.enable = angry
            bar._invoke([("power_on", None)], 0)
    finally:
        A._log_problem = saved
    assert logged and "power_on" in logged[0], f"nothing useful was logged: {logged!r}"


def test_the_icons_follow_the_daemon():
    with workdir("menubar-icons") as home:
        bar, daemon, _h, _c, status_bar = _bar(home)
        power_button = status_bar.items[0].button()
        dictation_button = status_bar.items[1].button()
        seen = {}
        for power in dictation.POWERS:
            for state in dictation.STATES:
                daemon.power, daemon.state = power, state
                bar.refresh()
                seen[(power, state)] = (power_button.title,
                                        dictation_button.title,
                                        dictation_button.dimmed)
        for (power, state), (power_title, dictation_title, dimmed) in seen.items():
            assert power_title == dictation.power_icon(power), \
                f"({power}, {state}) drew {power_title!r} for the power icon"
            expected_title, expected_dim = dictation.dictation_icon(power, state)
            assert dictation_title == expected_title, \
                f"({power}, {state}) drew {dictation_title!r}"
            assert dimmed is expected_dim, f"({power}, {state}) dimmed={dimmed}"
        print(f"  {len(seen)} combinations, all drawn from the model")


def test_refresh_redraws_only_when_something_changed():
    """Five wake-ups a second must not be five history reads a second."""
    with workdir("menubar-fingerprint") as home:
        bar, daemon, history, _c, _b = _bar(home)
        reads = []
        real_recent = history.recent
        history.recent = lambda *a, **kw: (reads.append(1), real_recent(*a, **kw))[1]
        bar.refresh()
        bar.refresh()
        bar.refresh()
        assert reads == [], f"an unchanged menu bar read the history {len(reads)} times"
        daemon.state = dictation.RECORDING
        bar.refresh()
        assert len(reads) == 1, f"a state change caused {len(reads)} reads"
        history.append("Bir cümle.", dictation.CORRECTED)
        bar.refresh()
        assert len(reads) == 2, \
            f"a new dictation was not noticed: {len(reads)} reads in total"
        print(f"  3 unchanged refreshes read nothing; 2 changes read {len(reads)} times")


def test_an_unreadable_history_does_not_look_empty():
    with workdir("menubar-unreadable") as home:
        path = pathlib.Path(home) / "dictations.jsonl"
        path.mkdir()                          # a directory where the log belongs
        logged = []
        saved = A._log_problem
        A._log_problem = lambda what, exc=None: logged.append(what)
        try:
            bar, _d, _h, _c, _b = _bar(home, history=dictation.History(path))
        finally:
            A._log_problem = saved
        titles = _titles(bar._dictation_menu)
        assert "Geçmiş okunamadı" in titles, titles
        assert "Henüz dictation yok" not in titles, titles
        assert logged, "the unreadable log was not reported anywhere"
        print(f"  {titles}")


def test_reveal_falls_back_to_the_folder_before_the_first_dictation():
    """`open -R` on a missing path exits non-zero and prints a message nobody would
    see from a menu (measured). Before the first dictation there is no log."""
    calls = []

    class FakeSubprocess:
        @staticmethod
        def run(argv, **_kw):
            calls.append(argv)

    saved = menubar.subprocess
    menubar.subprocess = FakeSubprocess
    try:
        with workdir("menubar-reveal") as home:
            bar, _d, history, _c, _b = _bar(home)
            bar._reveal()
            assert calls[-1] == ["open", str(history.path.parent)], calls[-1]
            history.append("Bir cümle.", dictation.CORRECTED)
            bar._reveal()
            assert calls[-1] == ["open", "-R", str(history.path)], calls[-1]
    finally:
        menubar.subprocess = saved
    print(f"  {calls}")


def test_the_heartbeat_interval_is_the_signal_latency_bound():
    """Not a behaviour test — a pin. The interval is not a preference: measured, a
    SIGUSR1 handler ran 4996 ms late with no timer scheduled and 197 ms with this one,
    so --toggle, --stop and Ctrl-C are all bounded by it."""
    assert 0 < menubar.HEARTBEAT_SECONDS <= 0.5, \
        f"HEARTBEAT_SECONDS is {menubar.HEARTBEAT_SECONDS}, which is a latency bound"


if __name__ == "__main__":
    run([n for n in sorted(globals()) if n.startswith("test_")], globals())
