"""
skin_gui.py — Terminal GUI for VGG19 Skin Disease Classifier
=============================================================
Run:
    python skin_gui.py

Controls:
    Tab / Arrow keys  — move between fields
    Enter             — confirm / run
    Space             — toggle checkbox
    Ctrl+C / Q        — quit
"""

import curses
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path



# ── Palette ───────────────────────────────────────────────────────────────────
C_BG        = 0   # terminal default
C_HEADER    = 1   # cyan bold
C_LABEL     = 2   # white
C_INPUT_OFF = 3   # dim input box
C_INPUT_ON  = 4   # active input box (green)
C_BTN_OFF   = 5   # dim button
C_BTN_ON    = 6   # active button (green bg)
C_OUTPUT    = 7   # output text
C_ERR       = 8   # red
C_SUCCESS   = 9   # bright green
C_BORDER    = 10  # dim border
C_BADGE     = 11  # magenta badge

def init_colors():
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(C_HEADER,    curses.COLOR_CYAN,    -1)
    curses.init_pair(C_LABEL,     curses.COLOR_WHITE,   -1)
    curses.init_pair(C_INPUT_OFF, curses.COLOR_WHITE,   curses.COLOR_BLACK)
    curses.init_pair(C_INPUT_ON,  curses.COLOR_GREEN,   curses.COLOR_BLACK)
    curses.init_pair(C_BTN_OFF,   curses.COLOR_BLACK,   curses.COLOR_WHITE)
    curses.init_pair(C_BTN_ON,    curses.COLOR_BLACK,   curses.COLOR_GREEN)
    curses.init_pair(C_OUTPUT,    curses.COLOR_WHITE,   -1)
    curses.init_pair(C_ERR,       curses.COLOR_RED,     -1)
    curses.init_pair(C_SUCCESS,   curses.COLOR_GREEN,   -1)
    curses.init_pair(C_BORDER,    curses.COLOR_WHITE,   -1)
    curses.init_pair(C_BADGE,     curses.COLOR_MAGENTA, -1)

# ── Helper: draw a box with a title ──────────────────────────────────────────
def draw_box(win, y, x, h, w, title="", color=C_BORDER):
    attr = curses.color_pair(color) | curses.A_DIM
    try:
        win.attron(attr)
        win.border()
        win.attroff(attr)
        if title:
            label = f" {title} "
            win.attron(curses.color_pair(C_HEADER) | curses.A_BOLD)
            win.addstr(0, 3, label)
            win.attroff(curses.color_pair(C_HEADER) | curses.A_BOLD)
    except curses.error:
        pass

# ── Input field ───────────────────────────────────────────────────────────────
class InputField:
    def __init__(self, label, placeholder="", value=""):
        self.label       = label
        self.placeholder = placeholder
        self.value       = value
        self.cursor      = len(value)

    def handle_key(self, key):
        if key in (curses.KEY_BACKSPACE, 127, 8):
            if self.cursor > 0:
                self.value  = self.value[:self.cursor-1] + self.value[self.cursor:]
                self.cursor -= 1
        elif key == curses.KEY_DC:
            self.value = self.value[:self.cursor] + self.value[self.cursor+1:]
        elif key == curses.KEY_LEFT:
            self.cursor = max(0, self.cursor - 1)
        elif key == curses.KEY_RIGHT:
            self.cursor = min(len(self.value), self.cursor + 1)
        elif key == curses.KEY_HOME:
            self.cursor = 0
        elif key == curses.KEY_END:
            self.cursor = len(self.value)
        elif 32 <= key <= 126:
            ch          = chr(key)
            self.value  = self.value[:self.cursor] + ch + self.value[self.cursor:]
            self.cursor += 1

    def draw(self, win, y, x, width, active):
        label_attr = curses.color_pair(C_LABEL) | curses.A_BOLD
        win.attron(label_attr)
        try: win.addstr(y, x, self.label + ":")
        except curses.error: pass
        win.attroff(label_attr)

        box_x   = x + len(self.label) + 2
        box_w   = max(4, width - len(self.label) - 3)
        display = self.value or self.placeholder
        is_ph   = not self.value
        attr    = curses.color_pair(C_INPUT_ON if active else C_INPUT_OFF)
        if is_ph:
            attr |= curses.A_DIM

        field_str = (display + " " * box_w)[:box_w]
        try:
            win.attron(attr)
            win.addstr(y, box_x, field_str)
            win.attroff(attr)
        except curses.error:
            pass

        if active and not is_ph:
            cur_x = box_x + min(self.cursor, box_w - 1)
            ch    = self.value[self.cursor] if self.cursor < len(self.value) else " "
            try:
                win.attron(curses.color_pair(C_INPUT_ON) | curses.A_REVERSE)
                win.addstr(y, cur_x, ch)
                win.attroff(curses.color_pair(C_INPUT_ON) | curses.A_REVERSE)
            except curses.error:
                pass

# ── Checkbox ──────────────────────────────────────────────────────────────────
class Checkbox:
    def __init__(self, label, checked=True):
        self.label   = label
        self.checked = checked

    def toggle(self):
        self.checked = not self.checked

    def draw(self, win, y, x, active):
        box   = "[✓]" if self.checked else "[ ]"
        color = curses.color_pair(C_INPUT_ON if active else C_LABEL)
        if active:
            color |= curses.A_BOLD
        try:
            win.attron(color)
            win.addstr(y, x, f"{box} {self.label}")
            win.attroff(color)
        except curses.error:
            pass

# ── Mode selector ─────────────────────────────────────────────────────────────
class ModeSelector:
    MODES = ["Single Image", "Directory / Dataset"]

    def __init__(self):
        self.selected = 0

    def draw(self, win, y, x, active):
        try:
            win.addstr(y, x, "Mode: ", curses.color_pair(C_LABEL) | curses.A_BOLD)
        except curses.error:
            pass
        cx = x + 7
        for i, m in enumerate(self.MODES):
            label = f" {m} "
            if i == self.selected:
                attr = curses.color_pair(C_BTN_ON) | curses.A_BOLD
            elif active:
                attr = curses.color_pair(C_BTN_OFF)
            else:
                attr = curses.color_pair(C_LABEL) | curses.A_DIM
            try:
                win.attron(attr)
                win.addstr(y, cx, label)
                win.attroff(attr)
            except curses.error:
                pass
            cx += len(label) + 1

    def handle_key(self, key):
        if key in (curses.KEY_LEFT, curses.KEY_RIGHT, ord('h'), ord('l')):
            self.selected = (self.selected + 1) % len(self.MODES)

# ── Main app ──────────────────────────────────────────────────────────────────
class App:
    FOCUS_MODEL    = 0
    FOCUS_PATH     = 1
    FOCUS_REPORT   = 2
    FOCUS_MODE     = 3
    FOCUS_TTA      = 4
    FOCUS_RUN      = 5
    FOCUS_QUIT     = 6
    N_FIELDS       = 7

    def __init__(self, stdscr):
        self.stdscr  = stdscr
        self.focus   = 0
        self.output  = []          # list of (text, color_pair)
        self.running = False

        # Widgets
        self.mode     = ModeSelector()
        self.f_model  = InputField("Model  ", "e.g. /home/user/skin_vgg19.keras")
        self.f_path   = InputField("Image  ", "e.g. /home/user/patient.jpg")
        self.f_report = InputField("Report ", "optional: results.csv  (dir mode only)")
        self.tta      = Checkbox("Enable Test-Time Augmentation (TTA ×7)", checked=True)

        init_colors()
        curses.curs_set(0)
        self.stdscr.nodelay(False)

    # ── Layout constants ──────────────────────────────────────────────────────
    def layout(self):
        H, W        = self.stdscr.getmaxyx()
        form_h      = 18
        form_w      = min(90, W - 4)
        form_y      = 2
        form_x      = (W - form_w) // 2
        out_y       = form_y + form_h + 1
        out_h       = max(6, H - out_y - 1)
        out_w       = form_w
        out_x       = form_x
        return H, W, form_y, form_x, form_h, form_w, out_y, out_x, out_h, out_w

    # ── Draw ──────────────────────────────────────────────────────────────────
    def draw(self):
        s = self.stdscr
        s.erase()
        H, W, fy, fx, fh, fw, oy, ox, oh, ow = self.layout()

        # ── Header bar ───────────────────────────────────────────────────────
        header = "  🔬  Skin Disease Classifier  "
        try:
            s.attron(curses.color_pair(C_HEADER) | curses.A_BOLD | curses.A_REVERSE)
            s.addstr(0, 0, " " * W)
            s.addstr(0, max(0, (W - len(header)) // 2), header)
            s.attroff(curses.color_pair(C_HEADER) | curses.A_BOLD | curses.A_REVERSE)
        except curses.error:
            pass

        # Classes badge
        classes = "Acne · Eczema · Fungal · Melanoma · Psoriasis · Vitiligo"
        try:
            s.attron(curses.color_pair(C_BADGE) | curses.A_DIM)
            s.addstr(1, max(0, (W - len(classes)) // 2), classes)
            s.attroff(curses.color_pair(C_BADGE) | curses.A_DIM)
        except curses.error:
            pass

        # ── Form box ─────────────────────────────────────────────────────────
        try:
            form_win = s.derwin(fh, fw, fy, fx)
            draw_box(form_win, 0, 0, fh, fw, "Configuration")
        except curses.error:
            form_win = s

        inner_x = 2
        row     = 2

        # Mode selector
        self.mode.draw(form_win, row, inner_x, self.focus == self.FOCUS_MODE)
        row += 2

        # Model path
        self.f_model.draw(form_win, row, inner_x, fw - 4, self.focus == self.FOCUS_MODEL)
        row += 2

        # Dynamic label based on mode
        mode_lbl = "Image  " if self.mode.selected == 0 else "Dir    "
        self.f_path.label = mode_lbl
        ph = "e.g. /home/user/patient.jpg" if self.mode.selected == 0 else "e.g. /home/user/data/test/"
        self.f_path.placeholder = ph
        self.f_path.draw(form_win, row, inner_x, fw - 4, self.focus == self.FOCUS_PATH)
        row += 2

        # Report (dim if single-image mode)
        rattr = curses.A_DIM if self.mode.selected == 0 else 0
        try:
            form_win.attron(rattr)
        except curses.error:
            pass
        self.f_report.draw(form_win, row, inner_x, fw - 4, self.focus == self.FOCUS_REPORT)
        try:
            form_win.attroff(rattr)
        except curses.error:
            pass
        row += 2

        # TTA checkbox
        self.tta.draw(form_win, row, inner_x, self.focus == self.FOCUS_TTA)
        row += 2

        # Buttons
        run_attr  = curses.color_pair(C_BTN_ON)  | curses.A_BOLD if self.focus == self.FOCUS_RUN  else curses.color_pair(C_BTN_OFF)
        quit_attr = curses.color_pair(C_ERR)      | curses.A_BOLD if self.focus == self.FOCUS_QUIT else curses.color_pair(C_LABEL) | curses.A_DIM
        run_lbl   = "  ▶  RUN  " if not self.running else "  ◌  RUNNING…  "
        try:
            form_win.attron(run_attr)
            form_win.addstr(row, inner_x, run_lbl)
            form_win.attroff(run_attr)
            form_win.attron(quit_attr)
            form_win.addstr(row, inner_x + len(run_lbl) + 2, "  ✕  QUIT  ")
            form_win.attroff(quit_attr)
        except curses.error:
            pass

        # Hint bar
        hint = "Tab/↑↓ navigate  |  Enter select  |  Space toggle  |  ←→ switch mode  |  Paste works in any field"
        try:
            form_win.attron(curses.color_pair(C_LABEL) | curses.A_DIM)
            form_win.addstr(fh - 2, inner_x, hint[:fw - 4])
            form_win.attroff(curses.color_pair(C_LABEL) | curses.A_DIM)
        except curses.error:
            pass

        # ── Output panel ─────────────────────────────────────────────────────
        try:
            out_win = s.derwin(oh, ow, oy, ox)
            draw_box(out_win, 0, 0, oh, ow, "Output")
            visible = self.output[-(oh - 2):]
            for i, (line, cp) in enumerate(visible):
                try:
                    out_win.attron(curses.color_pair(cp))
                    out_win.addstr(i + 1, 2, line[:ow - 4])
                    out_win.attroff(curses.color_pair(cp))
                except curses.error:
                    pass
        except curses.error:
            pass

        s.refresh()

    # ── Build & run command ───────────────────────────────────────────────────
    def build_cmd(self):
        script = Path(__file__).parent / "test_skin_classifier.py"
        model  = self.f_model.value.strip()
        path   = self.f_path.value.strip()

        errors = []
        if not model:
            errors.append("Model path is required.")
        elif not Path(model).exists():
            errors.append(f"Model not found: {model}")
        if not path:
            errors.append("Image / directory path is required.")
        elif not Path(path).exists():
            errors.append(f"Path not found: {path}")
        if not script.exists():
            errors.append(f"test_skin_classifier.py not found next to skin_gui.py")
        if errors:
            return None, errors

        cmd = [sys.executable, str(script), "--model", model]
        if self.mode.selected == 0:
            cmd += ["--image", path]
        else:
            cmd += ["--test_dir", path]
            report = self.f_report.value.strip()
            if report:
                cmd += ["--report", report]
        if not self.tta.checked:
            cmd.append("--no_tta")
        return cmd, []

    def run_inference(self):
        cmd, errors = self.build_cmd()
        if errors:
            for e in errors:
                self.output.append((f"✗  {e}", C_ERR))
            self.running = False
            self.draw()
            return

        self.output.append(("─" * 60, C_BORDER))
        self.output.append(("$ " + " ".join(cmd), C_LABEL))
        self.output.append(("", C_OUTPUT))
        self.draw()

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                cp = C_ERR if any(w in line.lower() for w in ("error", "warn", "✗")) else \
                     C_SUCCESS if any(w in line for w in ("✓", "Accuracy", "Prediction")) else \
                     C_OUTPUT
                self.output.append((line, cp))
                self.draw()
            proc.wait()
            self.output.append(("", C_OUTPUT))
            if proc.returncode == 0:
                self.output.append(("✓  Done.", C_SUCCESS))
            else:
                self.output.append((f"✗  Exited with code {proc.returncode}", C_ERR))
        except Exception as ex:
            self.output.append((f"✗  {ex}", C_ERR))
        finally:
            self.running = False
            self.draw()

    # ── Event loop ────────────────────────────────────────────────────────────
    def run(self):
        while True:
            self.draw()
            try:
                key = self.stdscr.getch()
            except KeyboardInterrupt:
                break

            # Global quit
            if key in (ord('q'), ord('Q')) and self.focus == self.FOCUS_QUIT:
                break
            if key == 3:   # Ctrl+C
                break

            # Tab / Shift-Tab navigation
            if key == ord('\t'):
                self.focus = (self.focus + 1) % self.N_FIELDS
                continue
            if key == curses.KEY_BTAB:
                self.focus = (self.focus - 1) % self.N_FIELDS
                continue
            if key == curses.KEY_DOWN:
                self.focus = (self.focus + 1) % self.N_FIELDS
                continue
            if key == curses.KEY_UP:
                self.focus = (self.focus - 1) % self.N_FIELDS
                continue

            # Enter
            if key in (curses.KEY_ENTER, 10, 13):
                if self.focus == self.FOCUS_RUN:
                    if not self.running:
                        self.running = True
                        t = threading.Thread(target=self.run_inference, daemon=True)
                        t.start()
                    continue
                if self.focus == self.FOCUS_QUIT:
                    break
                if self.focus == self.FOCUS_TTA:
                    self.tta.toggle()
                    continue
                if self.focus == self.FOCUS_MODE:
                    self.mode.handle_key(key)
                    continue
                # Move to next field on Enter in text fields
                self.focus = (self.focus + 1) % self.N_FIELDS
                continue

            # Space → toggle checkbox
            if key == ord(' '):
                if self.focus == self.FOCUS_TTA:
                    self.tta.toggle()
                continue

            # Route keys to active widget
            if self.focus == self.FOCUS_MODEL:
                self.f_model.handle_key(key)
            elif self.focus == self.FOCUS_PATH:
                self.f_path.handle_key(key)
            elif self.focus == self.FOCUS_REPORT:
                self.f_report.handle_key(key)
            elif self.focus == self.FOCUS_MODE:
                self.mode.handle_key(key)

# ── Entry point ───────────────────────────────────────────────────────────────
def main(stdscr):
    app = App(stdscr)
    app.run()

if __name__ == "__main__":
    curses.wrapper(main)
