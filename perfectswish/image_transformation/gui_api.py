def get_rect(image, AdjustmentApp, initial_rect=None):
    if initial_rect is None:
        initial_rect = [int(0.4 * x) for x in
                        [817, 324, 1186, 329, 1364, 836, 709, 831]]  # Initial rectangle coordinates
    current_rec = [None]  # something mutable
    try:
        def set_rect(cam_rect):
            current_rec[0] = cam_rect

        app = AdjustmentApp(image, set_rect, rect=initial_rect)
        app.root.mainloop()
    except ValueError as e:
        print(f"error: {e}")
    return current_rec[0]

