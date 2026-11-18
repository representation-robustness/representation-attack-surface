# Manual semantic audit sample

Seed 2026; 10 validly transformed test functions per dataset x family.
Mark each item: [x] semantics preserved, or [ ] and write the problem under it.

## devign / identifier / devign_01457

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,15 +1,15 @@
-static void fd_chr_update_read_handler(CharDriverState *chr)
+static void fd_chr_update_read_handler(CharDriverState *__v_0001)
 
 {
 
-    FDCharDriver *s = chr->opaque;
+    FDCharDriver *__v_0002 = __v_0001->opaque;
 
 
 
-    if (s->fd_in_tag) {
+    if (__v_0002->fd_in_tag) {
 
-        g_source_remove(s->fd_in_tag);
+        g_source_remove(__v_0002->fd_in_tag);
 
-        s->fd_in_tag = 0;
+        __v_0002->fd_in_tag = 0;
 
     }
@@ -17,7 +17,7 @@
 
 
-    if (s->fd_in) {
+    if (__v_0002->fd_in) {
 
-        s->fd_in_tag = io_add_watch_poll(s->fd_in, fd_chr_read_poll, fd_chr_read, chr);
+        __v_0002->fd_in_tag = io_add_watch_poll(__v_0002->fd_in, fd_chr_read_poll, fd_chr_read, __v_0001);
 
     }
```

## devign / identifier / devign_01487

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -3,27 +3,27 @@
 {
 
-    int nf;
+    int __v_0001;
 
-    Suite *s;
+    Suite *__v_0002;
 
-    SRunner *sr;
+    SRunner *__v_0003;
 
 
 
-    s = qfloat_suite();
+    __v_0002 = qfloat_suite();
 
-    sr = srunner_create(s);
+    __v_0003 = srunner_create(__v_0002);
 
 
 
-    srunner_run_all(sr, CK_NORMAL);
+    srunner_run_all(__v_0003, CK_NORMAL);
 
-    nf = srunner_ntests_failed(sr);
+    __v_0001 = srunner_ntests_failed(__v_0003);
 
-    srunner_free(sr);
+    srunner_free(__v_0003);
 
 
 
-    return (nf == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
+    return (__v_0001 == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
 
 }
```

## devign / identifier / devign_04196

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,43 +1,43 @@
-dshow_cycle_devices(AVFormatContext *avctx, ICreateDevEnum *devenum,
+dshow_cycle_devices(AVFormatContext *__v_0001, ICreateDevEnum *__v_0006,
 
-                    enum dshowDeviceType devtype, IBaseFilter **pfilter)
+                    enum dshowDeviceType __v_0010, IBaseFilter **__v_0013)
 
 {
 
-    struct dshow_ctx *ctx = avctx->priv_data;
+    struct dshow_ctx *__v_0005 = __v_0001->priv_data;
 
-    IBaseFilter *device_filter = NULL;
+    IBaseFilter *__v_0007 = NULL;
 
-    IEnumMoniker *classenum = NULL;
+    IEnumMoniker *__v_0004 = NULL;
 
-    IMoniker *m = NULL;
+    IMoniker *__v_0012 = NULL;
 
-    const char *device_name = ctx->device_name[devtype];
+    const char *__v_0009 = __v_0005->device_name[__v_0010];
 
-    int skip = (devtype == VideoDevice) ? ctx->video_device_number
+    int __v_0015 = (__v_0010 == VideoDevice) ? __v_0005->video_device_number
 
-                                        : ctx->audio_device_number;
+                                        : __v_0005->audio_device_number;
 
-    int r;
+    int __v_0014;
 
 
 
-    const GUID *device_guid[2] = { &CLSID_VideoInputDeviceCategory,
+    const GUID *__v_0008[2] = { &CLSID_VideoInputDeviceCategory,
 
                                    &CLSID_AudioInputDeviceCategory };
 
-    const char *devtypename = (devtype == VideoDevice) ? "video" : "audio";
+    const char *__v_0011 = (__v_0010 == VideoDevice) ? "video" : "audio";
 
 
 
-    r = ICreateDevEnum_CreateClassEnumerator(devenum, device_guid[devtype],
+    __v_0014 = ICreateDevEnum_CreateClassEnumerator(__v_0006, __v_0008[__v_0010],
 
-                                             (IEnumMoniker **) &classenum, 0);
+                                             (IEnumMoniker **) &__v_0004, 0);
 
-    if (r != S_OK) {
+    if (__v_0014 != S_OK) {
 
-        av_log(avctx, AV_LOG_ERROR, "Could not enumerate %s devices.\n",
+        av_log(__v_0001, AV_LOG_ERROR, "Could not enumerate %s devices.\n",
 
-               devtypename);
+               __v_0011);
 
         return AVERROR(EIO);
@@ -47,17 +47,17 @@
 
 
-    while (!device_filter && IEnumMoniker_Next(classenum, 1, &m, NULL) == S_OK) {
+    while (!__v_0007 && IEnumMoniker_Next(__v_0004, 1, &__v_0012, NULL) == S_OK) {
 
-        IPropertyBag *bag = NULL;
+        IPropertyBag *__v_0002 = NULL;
 
-        char *buf = NULL;
+        char *__v_0003 = NULL;
 
-        VARIANT var;
+        VARIANT __v_0016;
 
 
 
-        r = IMoniker_BindToStorage(m, 0, 0, &IID_IPropertyBag, (void *) &bag);
+        __v_0014 = IMoniker_BindToStorage(__v_0012, 0, 0, &IID_IPropertyBag, (void *) &__v_0002);
 
-        if (r != S_OK)
+        if (__v_0014 != S_OK)
 
             goto fail1;
@@ -65,9 +65,9 @@
 
 
-        var.vt = VT_BSTR;
+        __v_0016.vt = VT_BSTR;
 
-        r = IPropertyBag_Read(bag, L"FriendlyName", &var, NULL);
+        __v_0014 = IPropertyBag_Read(__v_0002, L"FriendlyName", &__v_0016, NULL);
 
-        if (r != S_OK)
+        if (__v_0014 != S_OK)
 
             goto fail1;
@@ -75,11 +75,11 @@
 
 
-        buf = dup_wchar_to_utf8(var.bstrVal);
+        __v_0003 = dup_wchar_to_utf8(__v_0016.bstrVal);
 
 
 
-        if (pfilter) {
+        if (__v_0013) {
 
-            if (strcmp(device_name, buf))
+            if (strcmp(__v_0009, __v_0003))
 
                 goto fail1;
@@ -87,11 +87,11 @@
 
 
-            if (!skip--)
+            if (!__v_0015--)
 
-                IMoniker_BindToObject(m, 0, 0, &IID_IBaseFilter, (void *) &device_filter);
+                IMoniker_BindToObject(__v_0012, 0, 0, &IID_IBaseFilter, (void *) &__v_0007);
 
         } else {
 
-            av_log(avctx, AV_LOG_INFO, " \"%s\"\n", buf);
+            av_log(__v_0001, AV_LOG_INFO, " \"%s\"\n", __v_0003);
 
         }
@@ -101,13 +101,13 @@
 fail1:
 
-        if (buf)
+        if (__v_0003)
 
-            av_free(buf);
+            av_free(__v_0003);
 
-        if (bag)
+        if (__v_0002)
 
-            IPropertyBag_Release(bag);
+            IPropertyBag_Release(__v_0002);
 
-        IMoniker_Release(m);
+        IMoniker_Release(__v_0012);
 
     }
@@ -115,15 +115,15 @@
 
 
-    IEnumMoniker_Release(classenum);
+    IEnumMoniker_Release(__v_0004);
 
 
 
-    if (pfilter) {
+    if (__v_0013) {
 
-        if (!device_filter) {
+        if (!__v_0007) {
 
-            av_log(avctx, AV_LOG_ERROR, "Could not find %s device.\n",
+            av_log(__v_0001, AV_LOG_ERROR, "Could not find %s device.\n",
 
-                   devtypename);
+                   __v_0011);
 
             return AVERROR(EIO);
@@ -131,5 +131,5 @@
         }
 
-        *pfilter = device_filter;
+        *__v_0013 = __v_0007;
 
     }
```

## devign / identifier / devign_07458

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,23 +1,23 @@
-static void migrate_set_downtime(QTestState *who, const char *value)
+static void migrate_set_downtime(QTestState *__v_0004, const char *__v_0003)
 
 {
 
-    QDict *rsp;
+    QDict *__v_0002;
 
-    gchar *cmd;
+    gchar *__v_0001;
 
 
 
-    cmd = g_strdup_printf("{ 'execute': 'migrate_set_downtime',"
+    __v_0001 = g_strdup_printf("{ 'execute': 'migrate_set_downtime',"
 
-                          "'arguments': { 'value': %s } }", value);
+                          "'arguments': { 'value': %s } }", __v_0003);
 
-    rsp = qtest_qmp(who, cmd);
+    __v_0002 = qtest_qmp(__v_0004, __v_0001);
 
-    g_free(cmd);
+    g_free(__v_0001);
 
-    g_assert(qdict_haskey(rsp, "return"));
+    g_assert(qdict_haskey(__v_0002, "return"));
 
-    QDECREF(rsp);
+    QDECREF(__v_0002);
 
 }
```

## devign / identifier / devign_09508

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,23 +1,23 @@
-static int qemu_rdma_register_and_get_keys(RDMAContext *rdma,
+static int qemu_rdma_register_and_get_keys(RDMAContext *__v_0008,
 
-        RDMALocalBlock *block, uintptr_t host_addr,
+        RDMALocalBlock *__v_0001, uintptr_t __v_0005,
 
-        uint32_t *lkey, uint32_t *rkey, int chunk,
+        uint32_t *__v_0007, uint32_t *__v_0009, int __v_0002,
 
-        uint8_t *chunk_start, uint8_t *chunk_end)
+        uint8_t *__v_0004, uint8_t *__v_0003)
 
 {
 
-    if (block->mr) {
+    if (__v_0001->mr) {
 
-        if (lkey) {
+        if (__v_0007) {
 
-            *lkey = block->mr->lkey;
+            *__v_0007 = __v_0001->mr->lkey;
 
         }
 
-        if (rkey) {
+        if (__v_0009) {
 
-            *rkey = block->mr->rkey;
+            *__v_0009 = __v_0001->mr->rkey;
 
         }
@@ -31,7 +31,7 @@
     /* allocate memory to store chunk MRs */
 
-    if (!block->pmr) {
+    if (!__v_0001->pmr) {
 
-        block->pmr = g_malloc0(block->nb_chunks * sizeof(struct ibv_mr *));
+        __v_0001->pmr = g_malloc0(__v_0001->nb_chunks * sizeof(struct ibv_mr *));
 
     }
@@ -49,19 +49,19 @@
      */
 
-    if (!block->pmr[chunk]) {
+    if (!__v_0001->pmr[__v_0002]) {
 
-        uint64_t len = chunk_end - chunk_start;
+        uint64_t __v_0006 = __v_0003 - __v_0004;
 
 
 
-        trace_qemu_rdma_register_and_get_keys(len, chunk_start);
+        trace_qemu_rdma_register_and_get_keys(__v_0006, __v_0004);
 
 
 
-        block->pmr[chunk] = ibv_reg_mr(rdma->pd,
+        __v_0001->pmr[__v_0002] = ibv_reg_mr(__v_0008->pd,
 
-                chunk_start, len,
+                __v_0004, __v_0006,
 
-                (rkey ? (IBV_ACCESS_LOCAL_WRITE |
+                (__v_0009 ? (IBV_ACCESS_LOCAL_WRITE |
 
                         IBV_ACCESS_REMOTE_WRITE) : 0));
@@ -69,5 +69,5 @@
 
 
-        if (!block->pmr[chunk]) {
+        if (!__v_0001->pmr[__v_0002]) {
 
             perror("Failed to register chunk!");
@@ -81,11 +81,11 @@
                             " local %" PRIuPTR " registrations: %d\n",
 
-                            block->index, chunk, (uintptr_t)chunk_start,
+                            __v_0001->index, __v_0002, (uintptr_t)__v_0004,
 
-                            (uintptr_t)chunk_end, host_addr,
+                            (uintptr_t)__v_0003, __v_0005,
 
-                            (uintptr_t)block->local_host_addr,
+                            (uintptr_t)__v_0001->local_host_addr,
 
-                            rdma->total_registrations);
+                            __v_0008->total_registrations);
 
             return -1;
@@ -93,5 +93,5 @@
         }
 
-        rdma->total_registrations++;
+        __v_0008->total_registrations++;
 
     }
@@ -99,13 +99,13 @@
 
 
-    if (lkey) {
+    if (__v_0007) {
 
-        *lkey = block->pmr[chunk]->lkey;
+        *__v_0007 = __v_0001->pmr[__v_0002]->lkey;
 
     }
 
-    if (rkey) {
+    if (__v_0009) {
 
-        *rkey = block->pmr[chunk]->rkey;
+        *__v_0009 = __v_0001->pmr[__v_0002]->rkey;
 
     }
```

## devign / identifier / devign_10640

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,9 +1,9 @@
-static inline int cris_addc_pi_m(int a, int **b)
+static inline int cris_addc_pi_m(int __v_0001, int **__v_0002)
 
 {
 
-	asm volatile ("addc [%1+], %0\n" : "+r" (a), "+b" (*b));
+	asm volatile ("addc [%1+], %0\n" : "+r" (__v_0001), "+b" (*__v_0002));
 
-	return a;
+	return __v_0001;
 
 }
```

## devign / identifier / devign_11167

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,3 +1,3 @@
-static void omap_rtc_reset(struct omap_rtc_s *s)
+static void omap_rtc_reset(struct omap_rtc_s *__v_0001)
 
 {
@@ -7,33 +7,33 @@
 
 
-    s->interrupts = 0;
+    __v_0001->interrupts = 0;
 
-    s->comp_reg = 0;
+    __v_0001->comp_reg = 0;
 
-    s->running = 0;
+    __v_0001->running = 0;
 
-    s->pm_am = 0;
+    __v_0001->pm_am = 0;
 
-    s->auto_comp = 0;
+    __v_0001->auto_comp = 0;
 
-    s->round = 0;
+    __v_0001->round = 0;
 
-    s->tick = qemu_get_clock(rt_clock);
+    __v_0001->tick = qemu_get_clock(rt_clock);
 
-    memset(&s->alarm_tm, 0, sizeof(s->alarm_tm));
+    memset(&__v_0001->alarm_tm, 0, sizeof(__v_0001->alarm_tm));
 
-    s->alarm_tm.tm_mday = 0x01;
+    __v_0001->alarm_tm.tm_mday = 0x01;
 
-    s->status = 1 << 7;
+    __v_0001->status = 1 << 7;
 
     qemu_get_timedate(&tm, 0);
 
-    s->ti = mktimegm(&tm);
+    __v_0001->ti = mktimegm(&tm);
 
 
 
-    omap_rtc_alarm_update(s);
+    omap_rtc_alarm_update(__v_0001);
 
-    omap_rtc_tick(s);
+    omap_rtc_tick(__v_0001);
 
 }
```

## devign / identifier / devign_13734

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,25 +1,25 @@
-static void test_validate_fail_struct_nested(TestInputVisitorData *data,
+static void test_validate_fail_struct_nested(TestInputVisitorData *__v_0001,
 
-                                              const void *unused)
+                                              const void *__v_0004)
 
 {
 
-    UserDefTwo *udp = NULL;
+    UserDefTwo *__v_0003 = NULL;
 
-    Error *err = NULL;
+    Error *__v_0002 = NULL;
 
-    Visitor *v;
+    Visitor *__v_0005;
 
 
 
-    v = validate_test_init(data, "{ 'string0': 'string0', 'dict1': { 'string1': 'string1', 'dict2': { 'userdef1': { 'integer': 42, 'string': 'string', 'extra': [42, 23, {'foo':'bar'}] }, 'string2': 'string2'}}}");
+    __v_0005 = validate_test_init(__v_0001, "{ 'string0': 'string0', 'dict1': { 'string1': 'string1', 'dict2': { 'userdef1': { 'integer': 42, 'string': 'string', 'extra': [42, 23, {'foo':'bar'}] }, 'string2': 'string2'}}}");
 
 
 
-    visit_type_UserDefTwo(v, NULL, &udp, &err);
+    visit_type_UserDefTwo(__v_0005, NULL, &__v_0003, &__v_0002);
 
-    error_free_or_abort(&err);
+    error_free_or_abort(&__v_0002);
 
-    qapi_free_UserDefTwo(udp);
+    qapi_free_UserDefTwo(__v_0003);
 
 }
```

## devign / identifier / devign_16614

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,31 +1,31 @@
-static int decode_0(PAFVideoDecContext *c, uint8_t *pkt, uint8_t code)
+static int decode_0(PAFVideoDecContext *__v_0002, uint8_t *__v_0017, uint8_t __v_0003)
 
 {
 
-    uint32_t opcode_size, offset;
-
-    uint8_t *dst, *dend, mask = 0, color = 0;
-
-    const uint8_t *src, *send, *opcodes;
-
-    int i, j, op = 0;
-
-
-
-    i = bytestream2_get_byte(&c->gb);
-
-    if (i) {
-
-        if (code & 0x10) {
-
-            int align;
-
-
-
-            align = bytestream2_tell(&c->gb) & 3;
-
-            if (align)
-
-                bytestream2_skip(&c->gb, 4 - align);
+    uint32_t __v_0014, __v_0011;
+
+    uint8_t *__v_0006, *__v_0005, __v_0010 = 0, __v_0004 = 0;
+
+    const uint8_t *__v_0019, *__v_0018, *__v_0015;
+
+    int __v_0007, __v_0008, __v_0012 = 0;
+
+
+
+    __v_0007 = bytestream2_get_byte(&__v_0002->gb);
+
+    if (__v_0007) {
+
+        if (__v_0003 & 0x10) {
+
+            int __v_0001;
+
+
+
+            __v_0001 = bytestream2_tell(&__v_0002->gb) & 3;
+
+            if (__v_0001)
+
+                bytestream2_skip(&__v_0002->gb, 4 - __v_0001);
 
         }
@@ -33,41 +33,41 @@
         do {
 
-            int page, val, x, y;
-
-            val    = bytestream2_get_be16(&c->gb);
-
-            page   = val >> 14;
-
-            x      = (val & 0x7F) * 2;
-
-            y      = ((val >> 7) & 0x7F) * 2;
-
-            dst    = c->frame[page] + x + y * c->width;
-
-            dend   = c->frame[page] + c->frame_size;
-
-            offset = (x & 0x7F) * 2;
-
-            j      = bytestream2_get_le16(&c->gb) + offset;
+            int __v_0016, __v_0020, __v_0021, __v_0022;
+
+            __v_0020    = bytestream2_get_be16(&__v_0002->gb);
+
+            __v_0016   = __v_0020 >> 14;
+
+            __v_0021      = (__v_0020 & 0x7F) * 2;
+
+            __v_0022      = ((__v_0020 >> 7) & 0x7F) * 2;
+
+            __v_0006    = __v_0002->frame[__v_0016] + __v_0021 + __v_0022 * __v_0002->width;
+
+            __v_0005   = __v_0002->frame[__v_0016] + __v_0002->frame_size;
+
+            __v_0011 = (__v_0021 & 0x7F) * 2;
+
+            __v_0008      = bytestream2_get_le16(&__v_0002->gb) + __v_0011;
 
             do {
 
-                offset++;
-
-                if (dst + 3 * c->width + 4 > dend)
+                __v_0011++;
+
+                if (__v_0006 + 3 * __v_0002->width + 4 > __v_0005)
 
                     return AVERROR_INVALIDDATA;
 
-                read4x4block(c, dst, c->width);
-
-                if ((offset & 0x3F) == 0)
-
-                    dst += c->width * 3;
-
-                dst += 4;
-
-            } while (offset < j);
-
-        } while (--i);
+                read4x4block(__v_0002, __v_0006, __v_0002->width);
+
+                if ((__v_0011 & 0x3F) == 0)
+
+                    __v_0006 += __v_0002->width * 3;
+
+                __v_0006 += 4;
+
+            } while (__v_0011 < __v_0008);
+
+        } while (--__v_0007);
 
     }
@@ -75,39 +75,39 @@
 
 
-    dst  = c->frame[c->current_frame];
-
-    dend = c->frame[c->current_frame] + c->frame_size;
+    __v_0006  = __v_0002->frame[__v_0002->current_frame];
+
+    __v_0005 = __v_0002->frame[__v_0002->current_frame] + __v_0002->frame_size;
 
     do {
 
-        set_src_position(c, &src, &send);
-
-        if ((src + 3 * c->width + 4 > send) ||
-
-            (dst + 3 * c->width + 4 > dend))
+        set_src_position(__v_0002, &__v_0019, &__v_0018);
+
+        if ((__v_0019 + 3 * __v_0002->width + 4 > __v_0018) ||
+
+            (__v_0006 + 3 * __v_0002->width + 4 > __v_0005))
 
             return AVERROR_INVALIDDATA;
 
-        copy_block4(dst, src, c->width, c->width, 4);
-
-        i++;
-
-        if ((i & 0x3F) == 0)
-
-            dst += c->width * 3;
-
-        dst += 4;
-
-    } while (i < c->video_size / 16);
-
-
-
-    opcode_size = bytestream2_get_le16(&c->gb);
-
-    bytestream2_skip(&c->gb, 2);
-
-
-
-    if (bytestream2_get_bytes_left(&c->gb) < opcode_size)
+        copy_block4(__v_0006, __v_0019, __v_0002->width, __v_0002->width, 4);
+
+        __v_0007++;
+
+        if ((__v_0007 & 0x3F) == 0)
+
+            __v_0006 += __v_0002->width * 3;
+
+        __v_0006 += 4;
+
+    } while (__v_0007 < __v_0002->video_size / 16);
+
+
+
+    __v_0014 = bytestream2_get_le16(&__v_0002->gb);
+
+    bytestream2_skip(&__v_0002->gb, 2);
+
+
+
+    if (bytestream2_get_bytes_left(&__v_0002->gb) < __v_0014)
 
         return AVERROR_INVALIDDATA;
@@ -115,33 +115,33 @@
 
 
-    opcodes = pkt + bytestream2_tell(&c->gb);
-
-    bytestream2_skipu(&c->gb, opcode_size);
-
-
-
-    dst = c->frame[c->current_frame];
-
-
-
-    for (i = 0; i < c->height; i += 4, dst += c->width * 3)
-
-        for (j = 0; j < c->width; j += 4, dst += 4) {
-
-            int opcode, k = 0;
-
-            if (op > opcode_size)
+    __v_0015 = __v_0017 + bytestream2_tell(&__v_0002->gb);
+
+    bytestream2_skipu(&__v_0002->gb, __v_0014);
+
+
+
+    __v_0006 = __v_0002->frame[__v_0002->current_frame];
+
+
+
+    for (__v_0007 = 0; __v_0007 < __v_0002->height; __v_0007 += 4, __v_0006 += __v_0002->width * 3)
+
+        for (__v_0008 = 0; __v_0008 < __v_0002->width; __v_0008 += 4, __v_0006 += 4) {
+
+            int __v_0013, __v_0009 = 0;
+
+            if (__v_0012 > __v_0014)
 
                 return AVERROR_INVALIDDATA;
 
-            if (j & 4) {
-
-                opcode = opcodes[op] & 15;
-
-                op++;
+            if (__v_0008 & 4) {
+
+                __v_0013 = __v_0015[__v_0012] & 15;
+
+                __v_0012++;
 
             } else {
 
-                opcode = opcodes[op] >> 4;
+                __v_0013 = __v_0015[__v_0012] >> 4;
 
             }
@@ -149,27 +149,27 @@
 
 
-            while (block_sequences[opcode][k]) {
-
-                offset = c->width * 2;
-
-                code   = block_sequences[opcode][k++];
-
-
-
-                switch (code) {
+            while (block_sequences[__v_0013][__v_0009]) {
+
+                __v_0011 = __v_0002->width * 2;
+
+                __v_0003   = block_sequences[__v_0013][__v_0009++];
+
+
+
+                switch (__v_0003) {
 
                 case 2:
 
-                    offset = 0;
+                    __v_0011 = 0;
 
                 case 3:
 
-                    color = bytestream2_get_byte(&c->gb);
+                    __v_0004 = bytestream2_get_byte(&__v_0002->gb);
 
                 case 4:
 
-                    mask = bytestream2_get_byte(&c->gb);
-
-                    copy_color_mask(dst + offset, c->width, mask, color);
+                    __v_0010 = bytestream2_get_byte(&__v_0002->gb);
+
+                    copy_color_mask(__v_0006 + __v_0011, __v_0002->width, __v_0010, __v_0004);
 
                     break;
@@ -177,19 +177,19 @@
                 case 5:
 
-                    offset = 0;
+                    __v_0011 = 0;
 
                 case 6:
 
-                    set_src_position(c, &src, &send);
+                    set_src_position(__v_0002, &__v_0019, &__v_0018);
 
                 case 7:
 
-                    if (src + offset + c->width + 4 > send)
+                    if (__v_0019 + __v_0011 + __v_0002->width + 4 > __v_0018)
 
                         return AVERROR_INVALIDDATA;
 
-                    mask = bytestream2_get_byte(&c->gb);
-
-                    copy_src_mask(dst + offset, c->width, mask, src + offset);
+                    __v_0010 = bytestream2_get_byte(&__v_0002->gb);
+
+                    copy_src_mask(__v_0006 + __v_0011, __v_0002->width, __v_0010, __v_0019 + __v_0011);
 
                     break;
```

## devign / identifier / devign_25182

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,40 +1,40 @@
-void sdl2_gl_scanout(DisplayChangeListener *dcl,
+void sdl2_gl_scanout(DisplayChangeListener *__v_0003,
 
-                     uint32_t backing_id, bool backing_y_0_top,
+                     uint32_t __v_0001, bool __v_0002,
 
 
-                     uint32_t x, uint32_t y,
+                     uint32_t __v_0007, uint32_t __v_0008,
 
-                     uint32_t w, uint32_t h)
+                     uint32_t __v_0006, uint32_t __v_0004)
 
 {
 
-    struct sdl2_console *scon = container_of(dcl, struct sdl2_console, dcl);
+    struct sdl2_console *__v_0005 = container_of(__v_0003, struct sdl2_console, __v_0003);
 
 
 
-    assert(scon->opengl);
+    assert(__v_0005->opengl);
 
-    scon->x = x;
+    __v_0005->x = __v_0007;
 
-    scon->y = y;
+    __v_0005->y = __v_0008;
 
-    scon->w = w;
+    __v_0005->w = __v_0006;
 
-    scon->h = h;
+    __v_0005->h = __v_0004;
 
-    scon->tex_id = backing_id;
+    __v_0005->tex_id = __v_0001;
 
-    scon->y0_top = backing_y_0_top;
+    __v_0005->y0_top = __v_0002;
 
 
 
-    SDL_GL_MakeCurrent(scon->real_window, scon->winctx);
+    SDL_GL_MakeCurrent(__v_0005->real_window, __v_0005->winctx);
 
 
 
-    if (scon->tex_id == 0 || scon->w == 0 || scon->h == 0) {
+    if (__v_0005->tex_id == 0 || __v_0005->w == 0 || __v_0005->h == 0) {
 
-        sdl2_set_scanout_mode(scon, false);
+        sdl2_set_scanout_mode(__v_0005, false);
 
         return;
@@ -44,9 +44,9 @@
 
 
-    sdl2_set_scanout_mode(scon, true);
+    sdl2_set_scanout_mode(__v_0005, true);
 
-    if (!scon->fbo_id) {
+    if (!__v_0005->fbo_id) {
 
-        glGenFramebuffers(1, &scon->fbo_id);
+        glGenFramebuffers(1, &__v_0005->fbo_id);
 
     }
@@ -54,9 +54,9 @@
 
 
-    glBindFramebuffer(GL_FRAMEBUFFER_EXT, scon->fbo_id);
+    glBindFramebuffer(GL_FRAMEBUFFER_EXT, __v_0005->fbo_id);
 
     glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
 
-                              GL_TEXTURE_2D, scon->tex_id, 0);
+                              GL_TEXTURE_2D, __v_0005->tex_id, 0);
 
 }
```

## devign / deadcode / devign_02950

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -3,4 +3,5 @@
 {
 
+    { int __dc_0001 = 0; if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
     MachineState *ms = MACHINE(obj);
 
@@ -20,4 +21,5 @@
                             machine_get_accel, machine_set_accel, NULL);
 
+    { int __dc_0002 = 0; if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
     object_property_set_description(obj, "accel",
 
@@ -54,4 +56,5 @@
                                     NULL);
 
+    { int __dm_0003 = 0; for (int __fl_0004 = 0; __fl_0004 < 0; __fl_0004++) { __dm_0003 += __fl_0004; } }
     object_property_add_str(obj, "kernel",
 
@@ -78,4 +81,5 @@
                             machine_get_append, machine_set_append, NULL);
 
+    { int __dc_0005 = 0; if (__dc_0005 != 0) { __dc_0005 = __dc_0005 + 1; } }
     object_property_set_description(obj, "append",
 
@@ -104,4 +108,5 @@
                                     NULL);
 
+    { int __dc_0006 = 0; if (__dc_0006 != 0) { __dc_0006 = __dc_0006 + 1; } }
     object_property_add(obj, "phandle-start", "int",
 
@@ -140,4 +145,5 @@
                              NULL);
 
+    { int __dm_0007 = 0; for (int __fl_0008 = 0; __fl_0008 < 0; __fl_0008++) { __dm_0007 += __fl_0008; } }
     object_property_set_description(obj, "dump-guest-core",
 
@@ -170,4 +176,5 @@
                                     NULL);
 
+    { int __dc_0009 = 0; if (__dc_0009 != 0) { __dc_0009 = __dc_0009 + 1; } }
     object_property_add_str(obj, "firmware",
 
```

## devign / deadcode / devign_05361

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -5,8 +5,10 @@
 {
 
+    { int __dc_0001 = 0; if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
     static int instance;
 
     char smb_conf[128];
 
+    { int __dc_0002 = 0; if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
     char smb_cmdline[128];
 
@@ -15,4 +17,5 @@
 
 
+    { int __dm_0003 = 0; for (int __fl_0004 = 0; __fl_0004 < 0; __fl_0004++) { __dm_0003 += __fl_0004; } }
     snprintf(s->smb_dir, sizeof(s->smb_dir), "/tmp/qemu-smb.%ld-%d",
 
@@ -21,4 +24,5 @@
     if (mkdir(s->smb_dir, 0700) < 0) {
 
+        { int __dc_0005 = 0; if (__dc_0005 != 0) { __dc_0005 = __dc_0005 + 1; } }
         config_error(mon, "could not create samba server dir '%s'\n",
 
@@ -29,4 +33,5 @@
     }
 
+    { int __dc_0006 = 0; if (__dc_0006 != 0) { __dc_0006 = __dc_0006 + 1; } }
     snprintf(smb_conf, sizeof(smb_conf), "%s/%s", s->smb_dir, "smb.conf");
 
@@ -35,8 +40,10 @@
     f = fopen(smb_conf, "w");
 
+    { int __dm_0007 = 0; for (int __fl_0008 = 0; __fl_0008 < 0; __fl_0008++) { __dm_0007 += __fl_0008; } }
     if (!f) {
 
         slirp_smb_cleanup(s);
 
+        { int __dc_0009 = 0; if (__dc_0009 != 0) { __dc_0009 = __dc_0009 + 1; } }
         config_error(mon, "could not create samba server "
 
```

## devign / deadcode / devign_07327

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -3,4 +3,5 @@
 {
 
+    { int __dc_0001 = 0; if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
     int ret = 0;
 
@@ -35,4 +36,5 @@
     offset_desc = s->offset_page;
 
+    { int __dc_0002 = 0; if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
     offset_data = offset_desc + sizeof(PageDescriptor) * s->num_dumpable;
 
@@ -85,4 +87,5 @@
     pd_zero.offset = cpu_convert_to_target64(offset_data, endian);
 
+    { int __dm_0003 = 0; for (int __fl_0004 = 0; __fl_0004 < 0; __fl_0004++) { __dm_0003 += __fl_0004; } }
     pd_zero.page_flags = cpu_convert_to_target64(0, endian);
 
@@ -125,4 +128,5 @@
                               false);
 
+            { int __dc_0005 = 0; if (__dc_0005 != 0) { __dc_0005 = __dc_0005 + 1; } }
             if (ret < 0) {
 
@@ -193,4 +197,5 @@
                     (size_out < s->page_size)) {
 
+                { int __dc_0006 = 0; if (__dc_0006 != 0) { __dc_0006 = __dc_0006 + 1; } }
                 pd.flags = cpu_convert_to_target32(DUMP_DH_COMPRESSED_LZO,
 
@@ -237,4 +242,5 @@
                     dump_error(s, "dump: failed to write page data.\n");
 
+                    { int __dm_0007 = 0; for (int __fl_0008 = 0; __fl_0008 < 0; __fl_0008++) { __dm_0007 += __fl_0008; } }
                     goto out;
 
@@ -285,4 +291,5 @@
 
 
+            { int __dc_0009 = 0; if (__dc_0009 != 0) { __dc_0009 = __dc_0009 + 1; } }
             ret = write_cache(&page_desc, &pd, sizeof(PageDescriptor), false);
 
```

## devign / deadcode / devign_14589

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -3,8 +3,11 @@
 {
 
+	{ int __dc_0001 = (flags ^ flags); if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
 	if (bs) {
 
+		{ int __dc_0002 = (flags ^ flags); if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
 		fprintf(stderr, "file open already, try 'help close'\n");
 
+		{ int __dm_0003 = 0; for (int __fl_0004 = 0; __fl_0004 < 0; __fl_0004++) { __dm_0003 += __fl_0004; } }
 		return 1;
 
@@ -13,6 +16,8 @@
 
 
+	{ int __dc_0005 = (flags ^ flags); if (__dc_0005 != 0) { __dc_0005 = __dc_0005 + 1; } }
 	bs = bdrv_new("hda");
 
+	{ int __dc_0006 = (flags ^ flags); if (__dc_0006 != 0) { __dc_0006 = __dc_0006 + 1; } }
 	if (!bs)
 
@@ -21,6 +26,8 @@
 
 
+	{ int __dm_0007 = 0; for (int __fl_0008 = 0; __fl_0008 < 0; __fl_0008++) { __dm_0007 += __fl_0008; } }
 	if (bdrv_open(bs, name, flags) == -1) {
 
+		{ int __dc_0009 = (flags ^ flags); if (__dc_0009 != 0) { __dc_0009 = __dc_0009 + 1; } }
 		fprintf(stderr, "%s: can't open device %s\n", progname, name);
 
```

## devign / deadcode / devign_14982

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -3,14 +3,18 @@
 {
 
+    { int __dc_0001 = (length ^ length); if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
     uint32_t msg_type;
 
+    { int __dc_0002 = (length ^ length); if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
     le32 *tmp = (le32 *) data;
 
 
 
+    { int __dm_0003 = 0; for (int __fl_0004 = 0; __fl_0004 < 0; __fl_0004++) { __dm_0003 += __fl_0004; } }
     msg_type = le32_to_cpup(tmp);
 
 
 
+    { int __dc_0005 = (length ^ length); if (__dc_0005 != 0) { __dc_0005 = __dc_0005 + 1; } }
     switch (msg_type) {
 
@@ -63,4 +67,5 @@
 
 
+    { int __dc_0006 = (length ^ length); if (__dc_0006 != 0) { __dc_0006 = __dc_0006 + 1; } }
     return USB_RET_STALL;
 
```

## devign / deadcode / devign_16792

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -3,8 +3,10 @@
 {
 
+    { int __dc_0001 = 0; if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
     PXA2xxState *cpu;
 
     DriveInfo *dinfo;
 
+    { int __dc_0002 = 0; if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
     int be;
 
@@ -13,4 +15,5 @@
 
 
+    { int __dm_0003 = 0; for (int __fl_0004 = 0; __fl_0004 < 0; __fl_0004++) { __dm_0003 += __fl_0004; } }
     uint32_t connex_rom = 0x01000000;
 
@@ -19,4 +22,5 @@
 
 
+    { int __dc_0005 = 0; if (__dc_0005 != 0) { __dc_0005 = __dc_0005 + 1; } }
     cpu = pxa255_init(address_space_mem, connex_ram);
 
@@ -25,4 +29,5 @@
     dinfo = drive_get(IF_PFLASH, 0, 0);
 
+    { int __dc_0006 = 0; if (__dc_0006 != 0) { __dc_0006 = __dc_0006 + 1; } }
     if (!dinfo && !qtest_enabled()) {
 
@@ -31,4 +36,5 @@
                 "'pflash' parameter\n");
 
+        { int __dm_0007 = 0; for (int __fl_0008 = 0; __fl_0008 < 0; __fl_0008++) { __dm_0007 += __fl_0008; } }
         exit(1);
 
@@ -55,4 +61,5 @@
                                2, 0, 0, 0, 0, be)) {
 
+        { int __dc_0009 = 0; if (__dc_0009 != 0) { __dc_0009 = __dc_0009 + 1; } }
         fprintf(stderr, "qemu: Error registering flash memory.\n");
 
```

## devign / deadcode / devign_19889

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -3,4 +3,5 @@
 {
 
+    { int __dc_0001 = 0; if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
     int ret;
 
@@ -13,4 +14,5 @@
     /* 1. streamoff */
 
+    { int __dc_0002 = 0; if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
     ret = ff_v4l2_context_set_status(&s->capture, VIDIOC_STREAMOFF);
 
@@ -29,4 +31,5 @@
      */
 
+    { int __dm_0003 = 0; for (int __fl_0004 = 0; __fl_0004 < 0; __fl_0004++) { __dm_0003 += __fl_0004; } }
     av_log(s->avctx, AV_LOG_DEBUG, "waiting for user to release AVBufferRefs\n");
 
@@ -37,4 +40,5 @@
 
 
+    { int __dc_0005 = 0; if (__dc_0005 != 0) { __dc_0005 = __dc_0005 + 1; } }
     ff_v4l2_context_release(&s->capture);
 
@@ -45,8 +49,10 @@
     ret = ff_v4l2_context_get_format(&s->capture);
 
+    { int __dc_0006 = 0; if (__dc_0006 != 0) { __dc_0006 = __dc_0006 + 1; } }
     if (ret) {
 
         av_log(s->avctx, AV_LOG_ERROR, "query the new capture format\n");
 
+        { int __dm_0007 = 0; for (int __fl_0008 = 0; __fl_0008 < 0; __fl_0008++) { __dm_0007 += __fl_0008; } }
         return ret;
 
@@ -59,4 +65,5 @@
     ret = ff_v4l2_context_set_format(&s->capture);
 
+    { int __dc_0009 = 0; if (__dc_0009 != 0) { __dc_0009 = __dc_0009 + 1; } }
     if (ret) {
 
```

## devign / deadcode / devign_19999

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -3,4 +3,5 @@
 {
 
+    { int __dc_0001 = 0; if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
     int i;
 
@@ -11,4 +12,5 @@
         /* NAA designator type */
 
+        { int __dc_0002 = 0; if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
         if (p[3] != 8) {
 
@@ -17,4 +19,5 @@
         }
 
+        { int __dm_0003 = 0; for (int __fl_0004 = 0; __fl_0004 < 0; __fl_0004++) { __dm_0003 += __fl_0004; } }
         *p_wwn = ldq_be_p(p + 4);
 
@@ -25,4 +28,5 @@
 
 
+    { int __dc_0005 = 0; if (__dc_0005 != 0) { __dc_0005 = __dc_0005 + 1; } }
     if ((p[1] & 0xF) == 8) {
 
@@ -31,4 +35,5 @@
         if (p[3] < 20 || memcmp(&p[4], "naa.", 4)) {
 
+            { int __dc_0006 = 0; if (__dc_0006 != 0) { __dc_0006 = __dc_0006 + 1; } }
             return -EINVAL;
 
@@ -37,4 +42,5 @@
         if (p[3] > 20 && p[24] != ',') {
 
+            { int __dm_0007 = 0; for (int __fl_0008 = 0; __fl_0008 < 0; __fl_0008++) { __dm_0007 += __fl_0008; } }
             return -EINVAL;
 
@@ -43,4 +49,5 @@
         *p_wwn = 0;
 
+        { int __dc_0009 = 0; if (__dc_0009 != 0) { __dc_0009 = __dc_0009 + 1; } }
         for (i = 8; i < 24; i++) {
 
```

## devign / deadcode / devign_23076

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -3,16 +3,21 @@
 {
 
+    { int __dc_0001 = 0; if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
     sPAPRDRConnector *drc = SPAPR_DR_CONNECTOR(d);
 
+    { int __dc_0002 = 0; if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
     sPAPRDRConnectorClass *drck = SPAPR_DR_CONNECTOR_GET_CLASS(drc);
 
 
 
+    { int __dm_0003 = 0; for (int __fl_0004 = 0; __fl_0004 < 0; __fl_0004++) { __dm_0003 += __fl_0004; } }
     trace_spapr_drc_reset(spapr_drc_index(drc));
 
 
 
+    { int __dc_0005 = 0; if (__dc_0005 != 0) { __dc_0005 = __dc_0005 + 1; } }
     g_free(drc->ccs);
 
+    { int __dc_0006 = 0; if (__dc_0006 != 0) { __dc_0006 = __dc_0006 + 1; } }
     drc->ccs = NULL;
 
@@ -31,6 +36,8 @@
      */
 
+    { int __dm_0007 = 0; for (int __fl_0008 = 0; __fl_0008 < 0; __fl_0008++) { __dm_0007 += __fl_0008; } }
     if (drc->awaiting_release) {
 
+        { int __dc_0009 = 0; if (__dc_0009 != 0) { __dc_0009 = __dc_0009 + 1; } }
         drck->set_isolation_state(drc, SPAPR_DR_ISOLATION_STATE_ISOLATED);
 
```

## devign / deadcode / devign_23786

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -3,4 +3,5 @@
                             AVPacket *avpkt) {
 
+    { int __dc_0001 = 0; if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
     const uint8_t *buf = avpkt->data;
 
@@ -31,4 +32,5 @@
     if (buf[0] != 0x0a || buf[1] > 5) {
 
+        { int __dc_0002 = 0; if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
         av_log(avctx, AV_LOG_ERROR, "this is not PCX encoded data\n");
 
@@ -67,4 +69,5 @@
 
 
+    { int __dm_0003 = 0; for (int __fl_0004 = 0; __fl_0004 < 0; __fl_0004++) { __dm_0003 += __fl_0004; } }
     bits_per_pixel     = buf[3];
 
@@ -141,4 +144,5 @@
         avcodec_set_dimensions(avctx, w, h);
 
+    { int __dc_0005 = 0; if (__dc_0005 != 0) { __dc_0005 = __dc_0005 + 1; } }
     if ((ret = avctx->get_buffer(avctx, p)) < 0) {
 
@@ -179,4 +183,5 @@
             for (x=0; x<w; x++) {
 
+                { int __dc_0006 = 0; if (__dc_0006 != 0) { __dc_0006 = __dc_0006 + 1; } }
                 ptr[3*x  ] = scanline[x                    ];
 
@@ -221,4 +226,5 @@
         if (*buf++ != 12) {
 
+            { int __dm_0007 = 0; for (int __fl_0008 = 0; __fl_0008 < 0; __fl_0008++) { __dm_0007 += __fl_0008; } }
             av_log(avctx, AV_LOG_ERROR, "expected palette after image data\n");
 
@@ -269,4 +275,5 @@
 
 
+            { int __dc_0009 = 0; if (__dc_0009 != 0) { __dc_0009 = __dc_0009 + 1; } }
             for (x=0; x<w; x++) {
 
```

## devign / controlflow / devign_01573

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -3,5 +3,5 @@
 {
 
-    while (sum>>16)
+    while ((sum>>16) && ((sum ^ sum) == 0))
 
 	sum = (sum & 0xFFFF)+(sum >> 16);
```

## devign / controlflow / devign_04781

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -27,11 +27,7 @@
     if (st) {
 
-        if (mov->per_stream_grouping)
-
-            group = st->index;
-
-        else
-
-            group = st->codecpar->codec_type;
+        { int __p_0001 = (mov->per_stream_grouping) ? 1 : 0;
+        if (__p_0001) { group = st->index; }
+        if (!__p_0001) { group = st->codecpar->codec_type; } }
 
 
@@ -67,7 +63,8 @@
     avio_w8(pb, version);
 
-    avio_wb24(pb, flags);
-
-    if (version == 1) {
+    { if ((1)) { avio_wb24(pb, flags); } }
+
+    { int __p_0002 = (version == 1) ? 1 : 0;
+    if (__p_0002) {
 
         avio_wb64(pb, track->time);
@@ -75,142 +72,139 @@
         avio_wb64(pb, track->time);
 
+    }
+    if (!__p_0002) {
+
+        avio_wb32(pb, track->time); /* creation time */
+
+        avio_wb32(pb, track->time); /* modification time */
+
+    } }
+
+    avio_wb32(pb, track->track_id); /* track-id */
+
+    { if ((1)) { avio_wb32(pb, 0); } } /* reserved */
+
+    if (!track->entry && mov->mode == MODE_ISM)
+
+        (version == 1) ? avio_wb64(pb, UINT64_C(0xffffffffffffffff)) : avio_wb32(pb, 0xffffffff);
+
+    else { int __p_0003 = (!track->entry) ? 1 : 0;
+    if (__p_0003) { (version == 1) ? avio_wb64(pb, 0) : avio_wb32(pb, 0); }
+    if (!__p_0003) { (version == 1) ? avio_wb64(pb, duration) : avio_wb32(pb, duration); } }
+
+
+
+    avio_wb32(pb, 0); /* reserved */
+
+    avio_wb32(pb, 0); /* reserved */
+
+    { if ((1)) { avio_wb16(pb, 0); } } /* layer */
+
+    avio_wb16(pb, group); /* alternate group) */
+
+    /* Volume, only for audio */
+
+    if (track->par->codec_type == AVMEDIA_TYPE_AUDIO)
+
+        avio_wb16(pb, 0x0100);
+
+    else
+
+        avio_wb16(pb, 0);
+
+    avio_wb16(pb, 0); /* reserved */
+
+
+
+    /* Matrix structure */
+
+
+    if (st && st->metadata) {
+
+        AVDictionaryEntry *rot = av_dict_get(st->metadata, "rotate", NULL, 0);
+
+        rotation = (rot && rot->value) ? atoi(rot->value) : 0;
+
+    }
+
+
+    if (display_matrix) {
+
+        for (i = 0; i < 9; i++)
+
+            avio_wb32(pb, display_matrix[i]);
+
+
+    } else if (rotation == 90) {
+
+        write_matrix(pb,  0,  1, -1,  0, track->par->height, 0);
+
+    } else if (rotation == 180) {
+
+        write_matrix(pb, -1,  0,  0, -1, track->par->width, track->par->height);
+
+    } else if (rotation == 270) {
+
+        write_matrix(pb,  0, -1,  1,  0, 0, track->par->width);
+
+
     } else {
 
-        avio_wb32(pb, track->time); /* creation time */
-
-        avio_wb32(pb, track->time); /* modification time */
-
-    }
-
-    avio_wb32(pb, track->track_id); /* track-id */
-
-    avio_wb32(pb, 0); /* reserved */
-
-    if (!track->entry && mov->mode == MODE_ISM)
-
-        (version == 1) ? avio_wb64(pb, UINT64_C(0xffffffffffffffff)) : avio_wb32(pb, 0xffffffff);
-
-    else if (!track->entry)
-
-        (version == 1) ? avio_wb64(pb, 0) : avio_wb32(pb, 0);
-
-    else
-
-        (version == 1) ? avio_wb64(pb, duration) : avio_wb32(pb, duration);
-
-
-
-    avio_wb32(pb, 0); /* reserved */
-
-    avio_wb32(pb, 0); /* reserved */
-
-    avio_wb16(pb, 0); /* layer */
-
-    avio_wb16(pb, group); /* alternate group) */
-
-    /* Volume, only for audio */
-
-    if (track->par->codec_type == AVMEDIA_TYPE_AUDIO)
-
-        avio_wb16(pb, 0x0100);
-
-    else
-
-        avio_wb16(pb, 0);
-
-    avio_wb16(pb, 0); /* reserved */
-
-
-
-    /* Matrix structure */
-
-
-    if (st && st->metadata) {
-
-        AVDictionaryEntry *rot = av_dict_get(st->metadata, "rotate", NULL, 0);
-
-        rotation = (rot && rot->value) ? atoi(rot->value) : 0;
-
-    }
-
-
-    if (display_matrix) {
-
-        for (i = 0; i < 9; i++)
-
-            avio_wb32(pb, display_matrix[i]);
-
-
-    } else if (rotation == 90) {
-
-        write_matrix(pb,  0,  1, -1,  0, track->par->height, 0);
-
-    } else if (rotation == 180) {
-
-        write_matrix(pb, -1,  0,  0, -1, track->par->width, track->par->height);
-
-    } else if (rotation == 270) {
-
-        write_matrix(pb,  0, -1,  1,  0, 0, track->par->width);
-
+        write_matrix(pb,  1,  0,  0,  1, 0, 0);
+
+    }
+
+    /* Track width and height, for visual only */
+
+    if (st && (track->par->codec_type == AVMEDIA_TYPE_VIDEO ||
+
+               track->par->codec_type == AVMEDIA_TYPE_SUBTITLE)) {
+
+        int64_t track_width_1616;
+
+        if (track->mode == MODE_MOV) {
+
+            track_width_1616 = track->par->width * 0x10000ULL;
+
+        } else {
+
+            track_width_1616 = av_rescale(st->sample_aspect_ratio.num,
+
+                                                  track->par->width * 0x10000LL,
+
+                                                  st->sample_aspect_ratio.den);
+
+            if (!track_width_1616 ||
+
+                track->height != track->par->height ||
+
+                track_width_1616 > UINT32_MAX)
+
+                track_width_1616 = track->par->width * 0x10000ULL;
+
+        }
+
+        if (track_width_1616 > UINT32_MAX) {
+
+            av_log(mov->fc, AV_LOG_WARNING, "track width is too large\n");
+
+            track_width_1616 = 0;
+
+        }
+
+        avio_wb32(pb, track_width_1616);
+
+        if (track->height > 0xFFFF) {
+
+            av_log(mov->fc, AV_LOG_WARNING, "track height is too large\n");
+
+            avio_wb32(pb, 0);
+
+        } else
+
+            avio_wb32(pb, track->height * 0x10000U);
 
     } else {
 
-        write_matrix(pb,  1,  0,  0,  1, 0, 0);
-
-    }
-
-    /* Track width and height, for visual only */
-
-    if (st && (track->par->codec_type == AVMEDIA_TYPE_VIDEO ||
-
-               track->par->codec_type == AVMEDIA_TYPE_SUBTITLE)) {
-
-        int64_t track_width_1616;
-
-        if (track->mode == MODE_MOV) {
-
-            track_width_1616 = track->par->width * 0x10000ULL;
-
-        } else {
-
-            track_width_1616 = av_rescale(st->sample_aspect_ratio.num,
-
-                                                  track->par->width * 0x10000LL,
-
-                                                  st->sample_aspect_ratio.den);
-
-            if (!track_width_1616 ||
-
-                track->height != track->par->height ||
-
-                track_width_1616 > UINT32_MAX)
-
-                track_width_1616 = track->par->width * 0x10000ULL;
-
-        }
-
-        if (track_width_1616 > UINT32_MAX) {
-
-            av_log(mov->fc, AV_LOG_WARNING, "track width is too large\n");
-
-            track_width_1616 = 0;
-
-        }
-
-        avio_wb32(pb, track_width_1616);
-
-        if (track->height > 0xFFFF) {
-
-            av_log(mov->fc, AV_LOG_WARNING, "track height is too large\n");
-
-            avio_wb32(pb, 0);
-
-        } else
-
-            avio_wb32(pb, track->height * 0x10000U);
-
-    } else {
-
         avio_wb32(pb, 0);
 
```

## devign / controlflow / devign_09761

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -125,11 +125,11 @@
 		register uint8_t x;
 
-		x          = src[i + 2];
-
-		dst[i + 1] = src[i + 1];
-
-		dst[i + 2] = src[i + 0];
-
-		dst[i + 0] = x;
+		{ int __st_0001 = 0;
+		while (__st_0001 < 4) {
+		    if (__st_0001 == 0) { x          = src[i + 2]; __st_0001 = 1; }
+		    else if (__st_0001 == 1) { dst[i + 1] = src[i + 1]; __st_0001 = 2; }
+		    else if (__st_0001 == 2) { dst[i + 2] = src[i + 0]; __st_0001 = 3; }
+		    else if (__st_0001 == 3) { dst[i + 0] = x; __st_0001 = 4; }
+		} }
 
 	}
```

## devign / controlflow / devign_10738

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -9,5 +9,5 @@
     unsigned ps, i, context_count;
 
-    memset(state, 128, sizeof(state));
+    { if ((1)) { memset(state, 128, sizeof(state)); } }
 
 
@@ -27,11 +27,11 @@
 
 
-    fs->slice_x      = get_symbol(c, state, 0) * f->width;
-
-    fs->slice_y      = get_symbol(c, state, 0) * f->height;
-
-    fs->slice_width  = (get_symbol(c, state, 0) + 1) * f->width + fs->slice_x;
-
-    fs->slice_height = (get_symbol(c, state, 0) + 1) * f->height + fs->slice_y;
+    { int __st_0001 = 0;
+    while (__st_0001 < 4) {
+        if (__st_0001 == 0) { fs->slice_x      = get_symbol(c, state, 0) * f->width; __st_0001 = 1; }
+        else if (__st_0001 == 1) { fs->slice_y      = get_symbol(c, state, 0) * f->height; __st_0001 = 2; }
+        else if (__st_0001 == 2) { fs->slice_width  = (get_symbol(c, state, 0) + 1) * f->width + fs->slice_x; __st_0001 = 3; }
+        else if (__st_0001 == 3) { fs->slice_height = (get_symbol(c, state, 0) + 1) * f->height + fs->slice_y; __st_0001 = 4; }
+    } }
 
 
@@ -83,5 +83,5 @@
         if (p->context_count < context_count) {
 
-            av_freep(&p->state);
+            { if ((1)) { av_freep(&p->state); } }
 
             av_freep(&p->vlc_state);
```

## devign / controlflow / devign_10773

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -7,17 +7,13 @@
 
 
-    dsputil_init(&s->dsp, avctx);
+    { if ((1)) { dsputil_init(&s->dsp, avctx); } }
 
-    avctx->coded_frame= (AVFrame*)&s->picture;
-
-
-
-    s->frame_width = avctx->width;
-
-    s->frame_height = avctx->height;
-
-
-
-    s->y_block_width = (s->frame_width + 15) / 16;
+    { int __st_0002 = 0;
+    while (__st_0002 < 4) {
+        if (__st_0002 == 0) { avctx->coded_frame= (AVFrame*)&s->picture; __st_0002 = 1; }
+        else if (__st_0002 == 1) { s->frame_width = avctx->width; __st_0002 = 2; }
+        else if (__st_0002 == 2) { s->frame_height = avctx->height; __st_0002 = 3; }
+        else if (__st_0002 == 3) { s->y_block_width = (s->frame_width + 15) / 16; __st_0002 = 4; }
+    } }
 
     s->y_block_height = (s->frame_height + 15) / 16;
@@ -46,5 +42,5 @@
     s->dummy          = av_mallocz((s->y_block_width+1)*s->y_block_height*sizeof(int32_t));
 
-    h263_encode_init(&s->m); //mv_penalty
+    { if ((1)) { h263_encode_init(&s->m); } } //mv_penalty
 
 
```

## devign / controlflow / devign_15368

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -17,5 +17,5 @@
 
 
-    g_strlcpy(combined_key, key, QIO_CHANNEL_WEBSOCK_CLIENT_KEY_LEN + 1);
+    { if ((1)) { g_strlcpy(combined_key, key, QIO_CHANNEL_WEBSOCK_CLIENT_KEY_LEN + 1); } }
 
     g_strlcat(combined_key, QIO_CHANNEL_WEBSOCK_GUID,
@@ -47,15 +47,15 @@
 
 
-    response = g_strdup_printf(QIO_CHANNEL_WEBSOCK_HANDSHAKE_RESPONSE, accept);
-
-    responselen = strlen(response);
-
-    buffer_reserve(&ioc->encoutput, responselen);
-
-    buffer_append(&ioc->encoutput, response, responselen);
+    { int __st_0001 = 0;
+    while (__st_0001 < 4) {
+        if (__st_0001 == 0) { response = g_strdup_printf(QIO_CHANNEL_WEBSOCK_HANDSHAKE_RESPONSE, accept); __st_0001 = 1; }
+        else if (__st_0001 == 1) { responselen = strlen(response); __st_0001 = 2; }
+        else if (__st_0001 == 2) { buffer_reserve(&ioc->encoutput, responselen); __st_0001 = 3; }
+        else if (__st_0001 == 3) { buffer_append(&ioc->encoutput, response, responselen); __st_0001 = 4; }
+    } }
 
 
 
-    g_free(accept);
+    { if ((1)) { g_free(accept); } }
 
     g_free(response);
```

## devign / controlflow / devign_16767

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -3,4 +3,5 @@
 {
 
+    { int __p_0001 = (0 == 0); if (__p_0001 && (__p_0001 == 1)) { __p_0001 = __p_0001; } }
     int64_t va = *(int64_t *)a, vb = *(int64_t *)b;
 
```

## devign / controlflow / devign_20282

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -3,4 +3,5 @@
 {
 
+    { int __p_0001 = (0 == 0); if (__p_0001 && (__p_0001 == 1)) { __p_0001 = __p_0001; } }
     val = cpu_to_be64(val);
 
```

## devign / controlflow / devign_21851

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -3,4 +3,5 @@
 {
 
+    { int __p_0001 = (0 == 0); if (__p_0001 && (__p_0001 == 1)) { __p_0001 = __p_0001; } }
     return quant_band_template(pvq, f, rc, band, X, Y, N, b, blocks, lowband, duration,
 
```

## devign / controlflow / devign_26480

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -3,4 +3,5 @@
 {
 
+    { int __p_0001 = (0 == 0); if (__p_0001 && (__p_0001 == 1)) { __p_0001 = __p_0001; } }
     env->fregs[f1].l.upper = float32_chs(env->fregs[f2].l.upper);
 
```

## diversevul / identifier / diversevul_test_000334

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,3 +1,3 @@
-int ImagingLibTiffEncode(Imaging im, ImagingCodecState state, UINT8* buffer, int bytes) {
+int ImagingLibTiffEncode(Imaging __v_0004, ImagingCodecState __v_0006, UINT8* __v_0001, int __v_0002) {
     /* One shot encoder. Encode everything to the tiff in the clientstate.
        If we're running off of a FD, then run once, we're good, everything
@@ -9,78 +9,78 @@
     */
 
-    TIFFSTATE *clientstate = (TIFFSTATE *)state->context;
-    TIFF *tiff = clientstate->tiff;
+    TIFFSTATE *__v_0003 = (TIFFSTATE *)__v_0006->context;
+    TIFF *__v_0007 = __v_0003->tiff;
 
-    TRACE(("in encoder: bytes %d\n", bytes));
-    TRACE(("State: count %d, state %d, x %d, y %d, ystep %d\n", state->count, state->state,
-           state->x, state->y, state->ystep));
-    TRACE(("State: xsize %d, ysize %d, xoff %d, yoff %d \n", state->xsize, state->ysize,
-           state->xoff, state->yoff));
-    TRACE(("State: bits %d, bytes %d \n", state->bits, state->bytes));
-    TRACE(("Buffer: %p: %c%c%c%c\n", buffer, (char)buffer[0], (char)buffer[1],(char)buffer[2], (char)buffer[3]));
-    TRACE(("State->Buffer: %c%c%c%c\n", (char)state->buffer[0], (char)state->buffer[1],(char)state->buffer[2], (char)state->buffer[3]));
+    TRACE(("in encoder: bytes %d\n", __v_0002));
+    TRACE(("State: count %d, state %d, x %d, y %d, ystep %d\n", __v_0006->count, __v_0006->state,
+           __v_0006->x, __v_0006->y, __v_0006->ystep));
+    TRACE(("State: xsize %d, ysize %d, xoff %d, yoff %d \n", __v_0006->xsize, __v_0006->ysize,
+           __v_0006->xoff, __v_0006->yoff));
+    TRACE(("State: bits %d, bytes %d \n", __v_0006->bits, __v_0006->bytes));
+    TRACE(("Buffer: %p: %c%c%c%c\n", __v_0001, (char)__v_0001[0], (char)__v_0001[1],(char)__v_0001[2], (char)__v_0001[3]));
+    TRACE(("State->Buffer: %c%c%c%c\n", (char)__v_0006->buffer[0], (char)__v_0006->buffer[1],(char)__v_0006->buffer[2], (char)__v_0006->buffer[3]));
     TRACE(("Image: mode %s, type %d, bands: %d, xsize %d, ysize %d \n",
-           im->mode, im->type, im->bands, im->xsize, im->ysize));
+           __v_0004->mode, __v_0004->type, __v_0004->bands, __v_0004->xsize, __v_0004->ysize));
     TRACE(("Image: image8 %p, image32 %p, image %p, block %p \n",
-           im->image8, im->image32, im->image, im->block));
+           __v_0004->image8, __v_0004->image32, __v_0004->image, __v_0004->block));
     TRACE(("Image: pixelsize: %d, linesize %d \n",
-           im->pixelsize, im->linesize));
+           __v_0004->pixelsize, __v_0004->linesize));
 
-    dump_state(clientstate);
+    dump_state(__v_0003);
 
-    if (state->state == 0) {
+    if (__v_0006->state == 0) {
         TRACE(("Encoding line bt line"));
-        while(state->y < state->ysize){
-            state->shuffle(state->buffer,
-                           (UINT8*) im->image[state->y + state->yoff] +
-                           state->xoff * im->pixelsize,
-                           state->xsize);
+        while(__v_0006->y < __v_0006->ysize){
+            __v_0006->shuffle(__v_0006->buffer,
+                           (UINT8*) __v_0004->image[__v_0006->y + __v_0006->yoff] +
+                           __v_0006->xoff * __v_0004->pixelsize,
+                           __v_0006->xsize);
 
-            if (TIFFWriteScanline(tiff, (tdata_t)(state->buffer), (uint32)state->y, 0) == -1) {
-                TRACE(("Encode Error, row %d\n", state->y));
-                state->errcode = IMAGING_CODEC_BROKEN;
-                TIFFClose(tiff);
-                if (!clientstate->fp){
-                    free(clientstate->data);
+            if (TIFFWriteScanline(__v_0007, (tdata_t)(__v_0006->buffer), (uint32)__v_0006->y, 0) == -1) {
+                TRACE(("Encode Error, row %d\n", __v_0006->y));
+                __v_0006->errcode = IMAGING_CODEC_BROKEN;
+                TIFFClose(__v_0007);
+                if (!__v_0003->fp){
+                    free(__v_0003->data);
                 }
                 return -1;
             }
-            state->y++;
+            __v_0006->y++;
         }
 
-        if (state->y == state->ysize) {
-            state->state=1;
+        if (__v_0006->y == __v_0006->ysize) {
+            __v_0006->state=1;
 
             TRACE(("Flushing \n"));
-            if (!TIFFFlush(tiff)) {
+            if (!TIFFFlush(__v_0007)) {
                 TRACE(("Error flushing the tiff"));
                 // likely reason is memory.
-                state->errcode = IMAGING_CODEC_MEMORY;
-                TIFFClose(tiff);
-                if (!clientstate->fp){
-                    free(clientstate->data);
+                __v_0006->errcode = IMAGING_CODEC_MEMORY;
+                TIFFClose(__v_0007);
+                if (!__v_0003->fp){
+                    free(__v_0003->data);
                 }
                 return -1;
             }
             TRACE(("Closing \n"));
-            TIFFClose(tiff);
+            TIFFClose(__v_0007);
             // reset the clientstate metadata to use it to read out the buffer.
-            clientstate->loc = 0;
-            clientstate->size = clientstate->eof; // redundant?
+            __v_0003->loc = 0;
+            __v_0003->size = __v_0003->eof; // redundant?
         }
     }
 
-    if (state->state == 1 && !clientstate->fp) {
-        int read = (int)_tiffReadProc(clientstate, (tdata_t)buffer, (tsize_t)bytes);
-        TRACE(("Buffer: %p: %c%c%c%c\n", buffer, (char)buffer[0], (char)buffer[1],(char)buffer[2], (char)buffer[3]));
-        if (clientstate->loc == clientstate->eof) {
+    if (__v_0006->state == 1 && !__v_0003->fp) {
+        int __v_0005 = (int)_tiffReadProc(__v_0003, (tdata_t)__v_0001, (tsize_t)__v_0002);
+        TRACE(("Buffer: %p: %c%c%c%c\n", __v_0001, (char)__v_0001[0], (char)__v_0001[1],(char)__v_0001[2], (char)__v_0001[3]));
+        if (__v_0003->loc == __v_0003->eof) {
             TRACE(("Hit EOF, calling an end, freeing data"));
-            state->errcode = IMAGING_CODEC_END;
-            free(clientstate->data);
+            __v_0006->errcode = IMAGING_CODEC_END;
+            free(__v_0003->data);
         }
-        return read;
+        return __v_0005;
     }
 
-    state->errcode = IMAGING_CODEC_END;
+    __v_0006->errcode = IMAGING_CODEC_END;
     return 0;
 }
```

## diversevul / identifier / diversevul_test_000392

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,7 +1,7 @@
-int dns_packet_is_reply_for(DnsPacket *p, const DnsResourceKey *key) {
-        int r;
+int dns_packet_is_reply_for(DnsPacket *__v_0002, const DnsResourceKey *__v_0001) {
+        int __v_0003;
 
-        assert(p);
-        assert(key);
+        assert(__v_0002);
+        assert(__v_0001);
 
         /* Checks if the specified packet is a reply for the specified
@@ -9,15 +9,15 @@
          * section. */
 
-        if (DNS_PACKET_QR(p) != 1)
+        if (DNS_PACKET_QR(__v_0002) != 1)
                 return 0;
 
         /* Let's unpack the packet, if that hasn't happened yet. */
-        r = dns_packet_extract(p);
-        if (r < 0)
-                return r;
+        __v_0003 = dns_packet_extract(__v_0002);
+        if (__v_0003 < 0)
+                return __v_0003;
 
-        if (p->question->n_keys != 1)
+        if (__v_0002->question->n_keys != 1)
                 return 0;
 
-        return dns_resource_key_equal(p->question->keys[0], key);
+        return dns_resource_key_equal(__v_0002->question->keys[0], __v_0001);
 }
```

## diversevul / identifier / diversevul_test_001493

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,7 +1,7 @@
-ok_error_t Binary::shift(size_t value) {
-  Header& header = this->header();
+ok_error_t Binary::shift(size_t __v_0005) {
+  Header& __v_0001 = this->header();
 
   // Offset of the load commands table
-  const uint64_t loadcommands_start = is64_ ? sizeof(details::mach_header_64) :
+  const uint64_t __v_0004 = is64_ ? sizeof(details::mach_header_64) :
                                               sizeof(details::mach_header);
 
@@ -17,25 +17,25 @@
   // |************************|
   // +------------------------+ <---------- __TEXT.end
-  const uint64_t loadcommands_end = loadcommands_start + header.sizeof_cmds();
+  const uint64_t __v_0003 = __v_0004 + __v_0001.sizeof_cmds();
 
   // Segment that wraps this load command table
-  SegmentCommand* load_cmd_segment = segment_from_offset(loadcommands_end);
-  if (load_cmd_segment == nullptr) {
+  SegmentCommand* __v_0002 = segment_from_offset(__v_0003);
+  if (__v_0002 == nullptr) {
     LIEF_WARN("Can't find segment associated with last load command");
     return make_error_code(lief_errors::file_format_error);
   }
   LIEF_DEBUG("LC Table wrapped by {} / End offset: 0x{:x} (size: {:x})",
-             load_cmd_segment->name(), loadcommands_end, load_cmd_segment->data_.size());
-  load_cmd_segment->content_insert(loadcommands_end, value);
+             __v_0002->name(), __v_0003, __v_0002->data_.size());
+  __v_0002->content_insert(__v_0003, __v_0005);
 
   // 1. Shift all commands
   // =====================
   for (std::unique_ptr<LoadCommand>& cmd : commands_) {
-    if (cmd->command_offset() >= loadcommands_end) {
-      cmd->command_offset(cmd->command_offset() + value);
+    if (cmd->command_offset() >= __v_0003) {
+      cmd->command_offset(cmd->command_offset() + __v_0005);
     }
   }
 
-  shift_command(value, loadcommands_end);
+  shift_command(__v_0005, __v_0003);
 
   // Shift Segment and sections
@@ -43,31 +43,31 @@
   for (SegmentCommand* segment : segments_) {
     // Extend the virtual size of the segment containing our shift
-    if (segment->file_offset() <= loadcommands_end &&
-        loadcommands_end < (segment->file_offset() + segment->file_size()))
+    if (segment->file_offset() <= __v_0003 &&
+        __v_0003 < (segment->file_offset() + segment->file_size()))
     {
-      LIEF_DEBUG("Extending '{}' by {:x}", segment->name(), value);
-      segment->virtual_size(segment->virtual_size() + value);
-      segment->file_size(segment->file_size() + value);
+      LIEF_DEBUG("Extending '{}' by {:x}", segment->name(), __v_0005);
+      segment->virtual_size(segment->virtual_size() + __v_0005);
+      segment->file_size(segment->file_size() + __v_0005);
 
       for (const std::unique_ptr<Section>& section : segment->sections_) {
-        if (section->offset() >= loadcommands_end) {
-          section->offset(section->offset() + value);
-          section->virtual_address(section->virtual_address() + value);
+        if (section->offset() >= __v_0003) {
+          section->offset(section->offset() + __v_0005);
+          section->virtual_address(section->virtual_address() + __v_0005);
         }
       }
     } else {
-      if (segment->file_offset() >= loadcommands_end) {
-        segment->file_offset(segment->file_offset() + value);
-        segment->virtual_address(segment->virtual_address() + value);
+      if (segment->file_offset() >= __v_0003) {
+        segment->file_offset(segment->file_offset() + __v_0005);
+        segment->virtual_address(segment->virtual_address() + __v_0005);
       }
 
       for (const std::unique_ptr<Section>& section : segment->sections_) {
-        if (section->offset() >= loadcommands_end) {
-          section->offset(section->offset() + value);
-          section->virtual_address(section->virtual_address() + value);
+        if (section->offset() >= __v_0003) {
+          section->offset(section->offset() + __v_0005);
+          section->virtual_address(section->virtual_address() + __v_0005);
         }
 
         if (section->type() == MACHO_SECTION_TYPES::S_ZEROFILL) {
-          section->virtual_address(section->virtual_address() + value);
+          section->virtual_address(section->virtual_address() + __v_0005);
         }
       }
```

## diversevul / identifier / diversevul_test_001652

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,28 +1,28 @@
-void usb_destroy_configuration(struct usb_device *dev)
+void usb_destroy_configuration(struct usb_device *__v_0003)
 {
-	int c, i;
+	int __v_0001, __v_0004;
 
-	if (!dev->config)
+	if (!__v_0003->config)
 		return;
 
-	if (dev->rawdescriptors) {
-		for (i = 0; i < dev->descriptor.bNumConfigurations; i++)
-			kfree(dev->rawdescriptors[i]);
+	if (__v_0003->rawdescriptors) {
+		for (__v_0004 = 0; __v_0004 < __v_0003->descriptor.bNumConfigurations; __v_0004++)
+			kfree(__v_0003->rawdescriptors[__v_0004]);
 
-		kfree(dev->rawdescriptors);
-		dev->rawdescriptors = NULL;
+		kfree(__v_0003->rawdescriptors);
+		__v_0003->rawdescriptors = NULL;
 	}
 
-	for (c = 0; c < dev->descriptor.bNumConfigurations; c++) {
-		struct usb_host_config *cf = &dev->config[c];
+	for (__v_0001 = 0; __v_0001 < __v_0003->descriptor.bNumConfigurations; __v_0001++) {
+		struct usb_host_config *__v_0002 = &__v_0003->config[__v_0001];
 
-		kfree(cf->string);
-		for (i = 0; i < cf->desc.bNumInterfaces; i++) {
-			if (cf->intf_cache[i])
-				kref_put(&cf->intf_cache[i]->ref,
+		kfree(__v_0002->string);
+		for (__v_0004 = 0; __v_0004 < __v_0002->desc.bNumInterfaces; __v_0004++) {
+			if (__v_0002->intf_cache[__v_0004])
+				kref_put(&__v_0002->intf_cache[__v_0004]->ref,
 					  usb_release_interface_cache);
 		}
 	}
-	kfree(dev->config);
-	dev->config = NULL;
+	kfree(__v_0003->config);
+	__v_0003->config = NULL;
 }
```

## diversevul / identifier / diversevul_test_002245

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,151 +1,151 @@
-PyString_Format(PyObject *format, PyObject *args)
+PyString_Format(PyObject *__v_0013, PyObject *__v_0004)
 {
-    char *fmt, *res;
-    Py_ssize_t arglen, argidx;
-    Py_ssize_t reslen, rescnt, fmtcnt;
-    int args_owned = 0;
-    PyObject *result, *orig_args;
-#ifdef Py_USING_UNICODE
-    PyObject *v, *w;
+    char *__v_0010, *__v_0027;
+    Py_ssize_t __v_0003, __v_0001;
+    Py_ssize_t __v_0029, __v_0028, __v_0012;
+    int __v_0005 = 0;
+    PyObject *__v_0030, *__v_0023;
+#ifdef Py_USING_UNICODE
+    PyObject *__v_0033, *__v_0034;
 #endif
-    PyObject *dict = NULL;
-    if (format == NULL || !PyString_Check(format) || args == NULL) {
+    PyObject *__v_0007 = NULL;
+    if (__v_0013 == NULL || !PyString_Check(__v_0013) || __v_0004 == NULL) {
         PyErr_BadInternalCall();
         return NULL;
     }
-    orig_args = args;
-    fmt = PyString_AS_STRING(format);
-    fmtcnt = PyString_GET_SIZE(format);
-    reslen = rescnt = fmtcnt + 100;
-    result = PyString_FromStringAndSize((char *)NULL, reslen);
-    if (result == NULL)
+    __v_0023 = __v_0004;
+    __v_0010 = PyString_AS_STRING(__v_0013);
+    __v_0012 = PyString_GET_SIZE(__v_0013);
+    __v_0029 = __v_0028 = __v_0012 + 100;
+    __v_0030 = PyString_FromStringAndSize((char *)NULL, __v_0029);
+    if (__v_0030 == NULL)
         return NULL;
-    res = PyString_AsString(result);
-    if (PyTuple_Check(args)) {
-        arglen = PyTuple_GET_SIZE(args);
-        argidx = 0;
+    __v_0027 = PyString_AsString(__v_0030);
+    if (PyTuple_Check(__v_0004)) {
+        __v_0003 = PyTuple_GET_SIZE(__v_0004);
+        __v_0001 = 0;
     }
     else {
-        arglen = -1;
-        argidx = -2;
-    }
-    if (Py_TYPE(args)->tp_as_mapping && Py_TYPE(args)->tp_as_mapping->mp_subscript &&
-        !PyTuple_Check(args) && !PyObject_TypeCheck(args, &PyBaseString_Type))
-        dict = args;
-    while (--fmtcnt >= 0) {
-        if (*fmt != '%') {
-            if (--rescnt < 0) {
-                rescnt = fmtcnt + 100;
-                reslen += rescnt;
-                if (_PyString_Resize(&result, reslen))
+        __v_0003 = -1;
+        __v_0001 = -2;
+    }
+    if (Py_TYPE(__v_0004)->tp_as_mapping && Py_TYPE(__v_0004)->tp_as_mapping->mp_subscript &&
+        !PyTuple_Check(__v_0004) && !PyObject_TypeCheck(__v_0004, &PyBaseString_Type))
+        __v_0007 = __v_0004;
+    while (--__v_0012 >= 0) {
+        if (*__v_0010 != '%') {
+            if (--__v_0028 < 0) {
+                __v_0028 = __v_0012 + 100;
+                __v_0029 += __v_0028;
+                if (_PyString_Resize(&__v_0030, __v_0029))
                     return NULL;
-                res = PyString_AS_STRING(result)
-                    + reslen - rescnt;
-                --rescnt;
-            }
-            *res++ = *fmt++;
+                __v_0027 = PyString_AS_STRING(__v_0030)
+                    + __v_0029 - __v_0028;
+                --__v_0028;
+            }
+            *__v_0027++ = *__v_0010++;
         }
         else {
             /* Got a format specifier */
-            int flags = 0;
-            Py_ssize_t width = -1;
-            int prec = -1;
-            int c = '\0';
-            int fill;
-            int isnumok;
-            PyObject *v = NULL;
-            PyObject *temp = NULL;
-            char *pbuf;
-            int sign;
-            Py_ssize_t len;
-            char formatbuf[FORMATBUFLEN];
+            int __v_0009 = 0;
+            Py_ssize_t __v_0035 = -1;
+            int __v_0026 = -1;
+            int __v_0006 = '\0';
+            int __v_0008;
+            int __v_0017;
+            PyObject *__v_0033 = NULL;
+            PyObject *__v_0032 = NULL;
+            char *__v_0024;
+            int __v_0031;
+            Py_ssize_t __v_0021;
+            char __v_0014[FORMATBUFLEN];
                  /* For format{int,char}() */
 #ifdef Py_USING_UNICODE
-            char *fmt_start = fmt;
-            Py_ssize_t argidx_start = argidx;
+            char *__v_0011 = __v_0010;
+            Py_ssize_t __v_0002 = __v_0001;
 #endif
 
-            fmt++;
-            if (*fmt == '(') {
-                char *keystart;
-                Py_ssize_t keylen;
-                PyObject *key;
-                int pcount = 1;
-
-                if (dict == NULL) {
+            __v_0010++;
+            if (*__v_0010 == '(') {
+                char *__v_0020;
+                Py_ssize_t __v_0019;
+                PyObject *__v_0018;
+                int __v_0025 = 1;
+
+                if (__v_0007 == NULL) {
                     PyErr_SetString(PyExc_TypeError,
                              "format requires a mapping");
                     goto error;
                 }
-                ++fmt;
-                --fmtcnt;
-                keystart = fmt;
+                ++__v_0010;
+                --__v_0012;
+                __v_0020 = __v_0010;
                 /* Skip over balanced parentheses */
-                while (pcount > 0 && --fmtcnt >= 0) {
-                    if (*fmt == ')')
-                        --pcount;
-                    else if (*fmt == '(')
-                        ++pcount;
-                    fmt++;
-                }
-                keylen = fmt - keystart - 1;
-                if (fmtcnt < 0 || pcount > 0) {
+                while (__v_0025 > 0 && --__v_0012 >= 0) {
+                    if (*__v_0010 == ')')
+                        --__v_0025;
+                    else if (*__v_0010 == '(')
+                        ++__v_0025;
+                    __v_0010++;
+                }
+                __v_0019 = __v_0010 - __v_0020 - 1;
+                if (__v_0012 < 0 || __v_0025 > 0) {
                     PyErr_SetString(PyExc_ValueError,
                                "incomplete format key");
                     goto error;
                 }
-                key = PyString_FromStringAndSize(keystart,
-                                                 keylen);
-                if (key == NULL)
-                    goto error;
-                if (args_owned) {
-                    Py_DECREF(args);
-                    args_owned = 0;
-                }
-                args = PyObject_GetItem(dict, key);
-                Py_DECREF(key);
-                if (args == NULL) {
-                    goto error;
-                }
-                args_owned = 1;
-                arglen = -1;
-                argidx = -2;
-            }
-            while (--fmtcnt >= 0) {
-                switch (c = *fmt++) {
-                case '-': flags |= F_LJUST; continue;
-                case '+': flags |= F_SIGN; continue;
-                case ' ': flags |= F_BLANK; continue;
-                case '#': flags |= F_ALT; continue;
-                case '0': flags |= F_ZERO; continue;
-                }
-                break;
-            }
-            if (c == '*') {
-                v = getnextarg(args, arglen, &argidx);
-                if (v == NULL)
-                    goto error;
-                if (!PyInt_Check(v)) {
+                __v_0018 = PyString_FromStringAndSize(__v_0020,
+                                                 __v_0019);
+                if (__v_0018 == NULL)
+                    goto error;
+                if (__v_0005) {
+                    Py_DECREF(__v_0004);
+                    __v_0005 = 0;
+                }
+                __v_0004 = PyObject_GetItem(__v_0007, __v_0018);
+                Py_DECREF(__v_0018);
+                if (__v_0004 == NULL) {
+                    goto error;
+                }
+                __v_0005 = 1;
+                __v_0003 = -1;
+                __v_0001 = -2;
+            }
+            while (--__v_0012 >= 0) {
+                switch (__v_0006 = *__v_0010++) {
+                case '-': __v_0009 |= F_LJUST; continue;
+                case '+': __v_0009 |= F_SIGN; continue;
+                case ' ': __v_0009 |= F_BLANK; continue;
+                case '#': __v_0009 |= F_ALT; continue;
+                case '0': __v_0009 |= F_ZERO; continue;
+                }
+                break;
+            }
+            if (__v_0006 == '*') {
+                __v_0033 = getnextarg(__v_0004, __v_0003, &__v_0001);
+                if (__v_0033 == NULL)
+                    goto error;
+                if (!PyInt_Check(__v_0033)) {
                     PyErr_SetString(PyExc_TypeError,
                                     "* wants int");
                     goto error;
                 }
-                width = PyInt_AsSsize_t(v);
-                if (width == -1 && PyErr_Occurred())
-                    goto error;
-                if (width < 0) {
-                    flags |= F_LJUST;
-                    width = -width;
-                }
-                if (--fmtcnt >= 0)
-                    c = *fmt++;
-            }
-            else if (c >= 0 && isdigit(c)) {
-                width = c - '0';
-                while (--fmtcnt >= 0) {
-                    c = Py_CHARMASK(*fmt++);
-                    if (!isdigit(c))
+                __v_0035 = PyInt_AsSsize_t(__v_0033);
+                if (__v_0035 == -1 && PyErr_Occurred())
+                    goto error;
+                if (__v_0035 < 0) {
+                    __v_0009 |= F_LJUST;
+                    __v_0035 = -__v_0035;
+                }
+                if (--__v_0012 >= 0)
+                    __v_0006 = *__v_0010++;
+            }
+            else if (__v_0006 >= 0 && isdigit(__v_0006)) {
+                __v_0035 = __v_0006 - '0';
+                while (--__v_0012 >= 0) {
+                    __v_0006 = Py_CHARMASK(*__v_0010++);
+                    if (!isdigit(__v_0006))
                         break;
-                    if (width > (PY_SSIZE_T_MAX - ((int)c - '0')) / 10) {
+                    if (__v_0035 > (PY_SSIZE_T_MAX - ((int)__v_0006 - '0')) / 10) {
                         PyErr_SetString(
                             PyExc_ValueError,
@@ -153,16 +153,16 @@
                         goto error;
                     }
-                    width = width*10 + (c - '0');
-                }
-            }
-            if (c == '.') {
-                prec = 0;
-                if (--fmtcnt >= 0)
-                    c = *fmt++;
-                if (c == '*') {
-                    v = getnextarg(args, arglen, &argidx);
-                    if (v == NULL)
+                    __v_0035 = __v_0035*10 + (__v_0006 - '0');
+                }
+            }
+            if (__v_0006 == '.') {
+                __v_0026 = 0;
+                if (--__v_0012 >= 0)
+                    __v_0006 = *__v_0010++;
+                if (__v_0006 == '*') {
+                    __v_0033 = getnextarg(__v_0004, __v_0003, &__v_0001);
+                    if (__v_0033 == NULL)
                         goto error;
-                    if (!PyInt_Check(v)) {
+                    if (!PyInt_Check(__v_0033)) {
                         PyErr_SetString(
                             PyExc_TypeError,
@@ -170,19 +170,19 @@
                         goto error;
                     }
-                    prec = _PyInt_AsInt(v);
-                    if (prec == -1 && PyErr_Occurred())
+                    __v_0026 = _PyInt_AsInt(__v_0033);
+                    if (__v_0026 == -1 && PyErr_Occurred())
                         goto error;
-                    if (prec < 0)
-                        prec = 0;
-                    if (--fmtcnt >= 0)
-                        c = *fmt++;
-                }
-                else if (c >= 0 && isdigit(c)) {
-                    prec = c - '0';
-                    while (--fmtcnt >= 0) {
-                        c = Py_CHARMASK(*fmt++);
-                        if (!isdigit(c))
+                    if (__v_0026 < 0)
+                        __v_0026 = 0;
+                    if (--__v_0012 >= 0)
+                        __v_0006 = *__v_0010++;
+                }
+                else if (__v_0006 >= 0 && isdigit(__v_0006)) {
+                    __v_0026 = __v_0006 - '0';
+                    while (--__v_0012 >= 0) {
+                        __v_0006 = Py_CHARMASK(*__v_0010++);
+                        if (!isdigit(__v_0006))
                             break;
-                        if (prec > (INT_MAX - ((int)c - '0')) / 10) {
+                        if (__v_0026 > (INT_MAX - ((int)__v_0006 - '0')) / 10) {
                             PyErr_SetString(
                                 PyExc_ValueError,
@@ -190,45 +190,45 @@
                             goto error;
                         }
-                        prec = prec*10 + (c - '0');
+                        __v_0026 = __v_0026*10 + (__v_0006 - '0');
                     }
                 }
             } /* prec */
-            if (fmtcnt >= 0) {
-                if (c == 'h' || c == 'l' || c == 'L') {
-                    if (--fmtcnt >= 0)
-                        c = *fmt++;
-                }
-            }
-            if (fmtcnt < 0) {
+            if (__v_0012 >= 0) {
+                if (__v_0006 == 'h' || __v_0006 == 'l' || __v_0006 == 'L') {
+                    if (--__v_0012 >= 0)
+                        __v_0006 = *__v_0010++;
+                }
+            }
+            if (__v_0012 < 0) {
                 PyErr_SetString(PyExc_ValueError,
                                 "incomplete format");
                 goto error;
             }
-            if (c != '%') {
-                v = getnextarg(args, arglen, &argidx);
-                if (v == NULL)
-                    goto error;
-            }
-            sign = 0;
-            fill = ' ';
-            switch (c) {
+            if (__v_0006 != '%') {
+                __v_0033 = getnextarg(__v_0004, __v_0003, &__v_0001);
+                if (__v_0033 == NULL)
+                    goto error;
+            }
+            __v_0031 = 0;
+            __v_0008 = ' ';
+            switch (__v_0006) {
             case '%':
-                pbuf = "%";
-                len = 1;
+                __v_0024 = "%";
+                __v_0021 = 1;
                 break;
             case 's':
 #ifdef Py_USING_UNICODE
-                if (PyUnicode_Check(v)) {
-                    fmt = fmt_start;
-                    argidx = argidx_start;
+                if (PyUnicode_Check(__v_0033)) {
+                    __v_0010 = __v_0011;
+                    __v_0001 = __v_0002;
                     goto unicode;
                 }
 #endif
-                temp = _PyObject_Str(v);
-#ifdef Py_USING_UNICODE
-                if (temp != NULL && PyUnicode_Check(temp)) {
-                    Py_DECREF(temp);
-                    fmt = fmt_start;
-                    argidx = argidx_start;
+                __v_0032 = _PyObject_Str(__v_0033);
+#ifdef Py_USING_UNICODE
+                if (__v_0032 != NULL && PyUnicode_Check(__v_0032)) {
+                    Py_DECREF(__v_0032);
+                    __v_0010 = __v_0011;
+                    __v_0001 = __v_0002;
                     goto unicode;
                 }
@@ -236,18 +236,18 @@
                 /* Fall through */
             case 'r':
-                if (c == 'r')
-                    temp = PyObject_Repr(v);
-                if (temp == NULL)
-                    goto error;
-                if (!PyString_Check(temp)) {
+                if (__v_0006 == 'r')
+                    __v_0032 = PyObject_Repr(__v_0033);
+                if (__v_0032 == NULL)
+                    goto error;
+                if (!PyString_Check(__v_0032)) {
                     PyErr_SetString(PyExc_TypeError,
                       "%s argument has non-string str()");
-                    Py_DECREF(temp);
-                    goto error;
-                }
-                pbuf = PyString_AS_STRING(temp);
-                len = PyString_GET_SIZE(temp);
-                if (prec >= 0 && len > prec)
-                    len = prec;
+                    Py_DECREF(__v_0032);
+                    goto error;
+                }
+                __v_0024 = PyString_AS_STRING(__v_0032);
+                __v_0021 = PyString_GET_SIZE(__v_0032);
+                if (__v_0026 >= 0 && __v_0021 > __v_0026)
+                    __v_0021 = __v_0026;
                 break;
             case 'i':
@@ -257,58 +257,58 @@
             case 'x':
             case 'X':
-                if (c == 'i')
-                    c = 'd';
-                isnumok = 0;
-                if (PyNumber_Check(v)) {
-                    PyObject *iobj=NULL;
-
-                    if (PyInt_Check(v) || (PyLong_Check(v))) {
-                        iobj = v;
-                        Py_INCREF(iobj);
+                if (__v_0006 == 'i')
+                    __v_0006 = 'd';
+                __v_0017 = 0;
+                if (PyNumber_Check(__v_0033)) {
+                    PyObject *__v_0016=NULL;
+
+                    if (PyInt_Check(__v_0033) || (PyLong_Check(__v_0033))) {
+                        __v_0016 = __v_0033;
+                        Py_INCREF(__v_0016);
                     }
                     else {
-                        iobj = PyNumber_Int(v);
-                        if (iobj==NULL) {
+                        __v_0016 = PyNumber_Int(__v_0033);
+                        if (__v_0016==NULL) {
                             PyErr_Clear();
-                            iobj = PyNumber_Long(v);
+                            __v_0016 = PyNumber_Long(__v_0033);
                         }
                     }
-                    if (iobj!=NULL) {
-                        if (PyInt_Check(iobj)) {
-                            isnumok = 1;
-                            pbuf = formatbuf;
-                            len = formatint(pbuf,
-                                            sizeof(formatbuf),
-                                            flags, prec, c, iobj);
-                            Py_DECREF(iobj);
-                            if (len < 0)
+                    if (__v_0016!=NULL) {
+                        if (PyInt_Check(__v_0016)) {
+                            __v_0017 = 1;
+                            __v_0024 = __v_0014;
+                            __v_0021 = formatint(__v_0024,
+                                            sizeof(__v_0014),
+                                            __v_0009, __v_0026, __v_0006, __v_0016);
+                            Py_DECREF(__v_0016);
+                            if (__v_0021 < 0)
                                 goto error;
-                            sign = 1;
+                            __v_0031 = 1;
                         }
-                        else if (PyLong_Check(iobj)) {
-                            int ilen;
-
-                            isnumok = 1;
-                            temp = _PyString_FormatLong(iobj, flags,
-                                prec, c, &pbuf, &ilen);
-                            Py_DECREF(iobj);
-                            len = ilen;
-                            if (!temp)
+                        else if (PyLong_Check(__v_0016)) {
+                            int __v_0015;
+
+                            __v_0017 = 1;
+                            __v_0032 = _PyString_FormatLong(__v_0016, __v_0009,
+                                __v_0026, __v_0006, &__v_0024, &__v_0015);
+                            Py_DECREF(__v_0016);
+                            __v_0021 = __v_0015;
+                            if (!__v_0032)
                                 goto error;
-                            sign = 1;
+                            __v_0031 = 1;
                         }
                         else {
-                            Py_DECREF(iobj);
+                            Py_DECREF(__v_0016);
                         }
                     }
                 }
-                if (!isnumok) {
+                if (!__v_0017) {
                     PyErr_Format(PyExc_TypeError,
                         "%%%c format: a number is required, "
-                        "not %.200s", c, Py_TYPE(v)->tp_name);
-                    goto error;
-                }
-                if (flags & F_ZERO)
-                    fill = '0';
+                        "not %.200s", __v_0006, Py_TYPE(__v_0033)->tp_name);
+                    goto error;
+                }
+                if (__v_0009 & F_ZERO)
+                    __v_0008 = '0';
                 break;
             case 'e':
@@ -318,24 +318,24 @@
             case 'g':
             case 'G':
-                temp = formatfloat(v, flags, prec, c);
-                if (temp == NULL)
-                    goto error;
-                pbuf = PyString_AS_STRING(temp);
-                len = PyString_GET_SIZE(temp);
-                sign = 1;
-                if (flags & F_ZERO)
-                    fill = '0';
+                __v_0032 = formatfloat(__v_0033, __v_0009, __v_0026, __v_0006);
+                if (__v_0032 == NULL)
+                    goto error;
+                __v_0024 = PyString_AS_STRING(__v_0032);
+                __v_0021 = PyString_GET_SIZE(__v_0032);
+                __v_0031 = 1;
+                if (__v_0009 & F_ZERO)
+                    __v_0008 = '0';
                 break;
             case 'c':
 #ifdef Py_USING_UNICODE
-                if (PyUnicode_Check(v)) {
-                    fmt = fmt_start;
-                    argidx = argidx_start;
+                if (PyUnicode_Check(__v_0033)) {
+                    __v_0010 = __v_0011;
+                    __v_0001 = __v_0002;
                     goto unicode;
                 }
 #endif
-                pbuf = formatbuf;
-                len = formatchar(pbuf, sizeof(formatbuf), v);
-                if (len < 0)
+                __v_0024 = __v_0014;
+                __v_0021 = formatchar(__v_0024, sizeof(__v_0014), __v_0033);
+                if (__v_0021 < 0)
                     goto error;
                 break;
@@ -344,155 +344,155 @@
                   "unsupported format character '%c' (0x%x) "
                   "at index %zd",
-                  c, c,
-                  (Py_ssize_t)(fmt - 1 -
-                               PyString_AsString(format)));
+                  __v_0006, __v_0006,
+                  (Py_ssize_t)(__v_0010 - 1 -
+                               PyString_AsString(__v_0013)));
                 goto error;
             }
-            if (sign) {
-                if (*pbuf == '-' || *pbuf == '+') {
-                    sign = *pbuf++;
-                    len--;
-                }
-                else if (flags & F_SIGN)
-                    sign = '+';
-                else if (flags & F_BLANK)
-                    sign = ' ';
+            if (__v_0031) {
+                if (*__v_0024 == '-' || *__v_0024 == '+') {
+                    __v_0031 = *__v_0024++;
+                    __v_0021--;
+                }
+                else if (__v_0009 & F_SIGN)
+                    __v_0031 = '+';
+                else if (__v_0009 & F_BLANK)
+                    __v_0031 = ' ';
                 else
-                    sign = 0;
-            }
-            if (width < len)
-                width = len;
-            if (rescnt - (sign != 0) < width) {
-                reslen -= rescnt;
-                rescnt = width + fmtcnt + 100;
-                reslen += rescnt;
-                if (reslen < 0) {
-                    Py_DECREF(result);
-                    Py_XDECREF(temp);
+                    __v_0031 = 0;
+            }
+            if (__v_0035 < __v_0021)
+                __v_0035 = __v_0021;
+            if (__v_0028 - (__v_0031 != 0) < __v_0035) {
+                __v_0029 -= __v_0028;
+                __v_0028 = __v_0035 + __v_0012 + 100;
+                __v_0029 += __v_0028;
+                if (__v_0029 < 0) {
+                    Py_DECREF(__v_0030);
+                    Py_XDECREF(__v_0032);
                     return PyErr_NoMemory();
                 }
-                if (_PyString_Resize(&result, reslen)) {
-                    Py_XDECREF(temp);
+                if (_PyString_Resize(&__v_0030, __v_0029)) {
+                    Py_XDECREF(__v_0032);
                     return NULL;
                 }
-                res = PyString_AS_STRING(result)
-                    + reslen - rescnt;
-            }
-            if (sign) {
-                if (fill != ' ')
-                    *res++ = sign;
-                rescnt--;
-                if (width > len)
-                    width--;
-            }
-            if ((flags & F_ALT) && (c == 'x' || c == 'X')) {
-                assert(pbuf[0] == '0');
-                assert(pbuf[1] == c);
-                if (fill != ' ') {
-                    *res++ = *pbuf++;
-                    *res++ = *pbuf++;
-                }
-                rescnt -= 2;
-                width -= 2;
-                if (width < 0)
-                    width = 0;
-                len -= 2;
-            }
-            if (width > len && !(flags & F_LJUST)) {
+                __v_0027 = PyString_AS_STRING(__v_0030)
+                    + __v_0029 - __v_0028;
+            }
+            if (__v_0031) {
+                if (__v_0008 != ' ')
+                    *__v_0027++ = __v_0031;
+                __v_0028--;
+                if (__v_0035 > __v_0021)
+                    __v_0035--;
+            }
+            if ((__v_0009 & F_ALT) && (__v_0006 == 'x' || __v_0006 == 'X')) {
+                assert(__v_0024[0] == '0');
+                assert(__v_0024[1] == __v_0006);
+                if (__v_0008 != ' ') {
+                    *__v_0027++ = *__v_0024++;
+                    *__v_0027++ = *__v_0024++;
+                }
+                __v_0028 -= 2;
+                __v_0035 -= 2;
+                if (__v_0035 < 0)
+                    __v_0035 = 0;
+                __v_0021 -= 2;
+            }
+            if (__v_0035 > __v_0021 && !(__v_0009 & F_LJUST)) {
                 do {
-                    --rescnt;
-                    *res++ = fill;
-                } while (--width > len);
-            }
-            if (fill == ' ') {
-                if (sign)
-                    *res++ = sign;
-                if ((flags & F_ALT) &&
-                    (c == 'x' || c == 'X')) {
-                    assert(pbuf[0] == '0');
-                    assert(pbuf[1] == c);
-                    *res++ = *pbuf++;
-                    *res++ = *pbuf++;
-                }
-            }
-            Py_MEMCPY(res, pbuf, len);
-            res += len;
-            rescnt -= len;
-            while (--width >= len) {
-                --rescnt;
-                *res++ = ' ';
-            }
-            if (dict && (argidx < arglen) && c != '%') {
+                    --__v_0028;
+                    *__v_0027++ = __v_0008;
+                } while (--__v_0035 > __v_0021);
+            }
+            if (__v_0008 == ' ') {
+                if (__v_0031)
+                    *__v_0027++ = __v_0031;
+                if ((__v_0009 & F_ALT) &&
+                    (__v_0006 == 'x' || __v_0006 == 'X')) {
+                    assert(__v_0024[0] == '0');
+                    assert(__v_0024[1] == __v_0006);
+                    *__v_0027++ = *__v_0024++;
+                    *__v_0027++ = *__v_0024++;
+                }
+            }
+            Py_MEMCPY(__v_0027, __v_0024, __v_0021);
+            __v_0027 += __v_0021;
+            __v_0028 -= __v_0021;
+            while (--__v_0035 >= __v_0021) {
+                --__v_0028;
+                *__v_0027++ = ' ';
+            }
+            if (__v_0007 && (__v_0001 < __v_0003) && __v_0006 != '%') {
                 PyErr_SetString(PyExc_TypeError,
                            "not all arguments converted during string formatting");
-                Py_XDECREF(temp);
+                Py_XDECREF(__v_0032);
                 goto error;
             }
-            Py_XDECREF(temp);
+            Py_XDECREF(__v_0032);
         } /* '%' */
     } /* until end */
-    if (argidx < arglen && !dict) {
+    if (__v_0001 < __v_0003 && !__v_0007) {
         PyErr_SetString(PyExc_TypeError,
                         "not all arguments converted during string formatting");
         goto error;
     }
-    if (args_owned) {
-        Py_DECREF(args);
-    }
-    if (_PyString_Resize(&result, reslen - rescnt))
+    if (__v_0005) {
+        Py_DECREF(__v_0004);
+    }
+    if (_PyString_Resize(&__v_0030, __v_0029 - __v_0028))
         return NULL;
-    return result;
+    return __v_0030;
 
 #ifdef Py_USING_UNICODE
  unicode:
-    if (args_owned) {
-        Py_DECREF(args);
-        args_owned = 0;
+    if (__v_0005) {
+        Py_DECREF(__v_0004);
+        __v_0005 = 0;
     }
     /* Fiddle args right (remove the first argidx arguments) */
-    if (PyTuple_Check(orig_args) && argidx > 0) {
-        PyObject *v;
-        Py_ssize_t n = PyTuple_GET_SIZE(orig_args) - argidx;
-        v = PyTuple_New(n);
-        if (v == NULL)
+    if (PyTuple_Check(__v_0023) && __v_0001 > 0) {
+        PyObject *__v_0033;
+        Py_ssize_t __v_0022 = PyTuple_GET_SIZE(__v_0023) - __v_0001;
+        __v_0033 = PyTuple_New(__v_0022);
+        if (__v_0033 == NULL)
             goto error;
-        while (--n >= 0) {
-            PyObject *w = PyTuple_GET_ITEM(orig_args, n + argidx);
-            Py_INCREF(w);
-            PyTuple_SET_ITEM(v, n, w);
+        while (--__v_0022 >= 0) {
+            PyObject *__v_0034 = PyTuple_GET_ITEM(__v_0023, __v_0022 + __v_0001);
+            Py_INCREF(__v_0034);
+            PyTuple_SET_ITEM(__v_0033, __v_0022, __v_0034);
         }
-        args = v;
+        __v_0004 = __v_0033;
     } else {
-        Py_INCREF(orig_args);
-        args = orig_args;
-    }
-    args_owned = 1;
+        Py_INCREF(__v_0023);
+        __v_0004 = __v_0023;
+    }
+    __v_0005 = 1;
     /* Take what we have of the result and let the Unicode formatting
        function format the rest of the input. */
-    rescnt = res - PyString_AS_STRING(result);
-    if (_PyString_Resize(&result, rescnt))
+    __v_0028 = __v_0027 - PyString_AS_STRING(__v_0030);
+    if (_PyString_Resize(&__v_0030, __v_0028))
         goto error;
-    fmtcnt = PyString_GET_SIZE(format) - \
-             (fmt - PyString_AS_STRING(format));
-    format = PyUnicode_Decode(fmt, fmtcnt, NULL, NULL);
-    if (format == NULL)
+    __v_0012 = PyString_GET_SIZE(__v_0013) - \
+             (__v_0010 - PyString_AS_STRING(__v_0013));
+    __v_0013 = PyUnicode_Decode(__v_0010, __v_0012, NULL, NULL);
+    if (__v_0013 == NULL)
         goto error;
-    v = PyUnicode_Format(format, args);
-    Py_DECREF(format);
-    if (v == NULL)
+    __v_0033 = PyUnicode_Format(__v_0013, __v_0004);
+    Py_DECREF(__v_0013);
+    if (__v_0033 == NULL)
         goto error;
     /* Paste what we have (result) to what the Unicode formatting
        function returned (v) and return the result (or error) */
-    w = PyUnicode_Concat(result, v);
-    Py_DECREF(result);
-    Py_DECREF(v);
-    Py_DECREF(args);
-    return w;
+    __v_0034 = PyUnicode_Concat(__v_0030, __v_0033);
+    Py_DECREF(__v_0030);
+    Py_DECREF(__v_0033);
+    Py_DECREF(__v_0004);
+    return __v_0034;
 #endif /* Py_USING_UNICODE */
 
  error:
-    Py_DECREF(result);
-    if (args_owned) {
-        Py_DECREF(args);
+    Py_DECREF(__v_0030);
+    if (__v_0005) {
+        Py_DECREF(__v_0004);
     }
     return NULL;
```

## diversevul / identifier / diversevul_test_002431

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,28 +1,28 @@
-usm_free_usmStateReference(void *old)
+usm_free_usmStateReference(void *__v_0001)
 {
-    struct usmStateReference *old_ref = (struct usmStateReference *) old;
+    struct usmStateReference *__v_0002 = (struct usmStateReference *) __v_0001;
 
-    if (old_ref) {
+    if (__v_0002) {
 
-        if (old_ref->usr_name_length)
-            SNMP_FREE(old_ref->usr_name);
-        if (old_ref->usr_engine_id_length)
-            SNMP_FREE(old_ref->usr_engine_id);
-        if (old_ref->usr_auth_protocol_length)
-            SNMP_FREE(old_ref->usr_auth_protocol);
-        if (old_ref->usr_priv_protocol_length)
-            SNMP_FREE(old_ref->usr_priv_protocol);
+        if (__v_0002->usr_name_length)
+            SNMP_FREE(__v_0002->usr_name);
+        if (__v_0002->usr_engine_id_length)
+            SNMP_FREE(__v_0002->usr_engine_id);
+        if (__v_0002->usr_auth_protocol_length)
+            SNMP_FREE(__v_0002->usr_auth_protocol);
+        if (__v_0002->usr_priv_protocol_length)
+            SNMP_FREE(__v_0002->usr_priv_protocol);
 
-        if (old_ref->usr_auth_key_length && old_ref->usr_auth_key) {
-            SNMP_ZERO(old_ref->usr_auth_key, old_ref->usr_auth_key_length);
-            SNMP_FREE(old_ref->usr_auth_key);
+        if (__v_0002->usr_auth_key_length && __v_0002->usr_auth_key) {
+            SNMP_ZERO(__v_0002->usr_auth_key, __v_0002->usr_auth_key_length);
+            SNMP_FREE(__v_0002->usr_auth_key);
         }
-        if (old_ref->usr_priv_key_length && old_ref->usr_priv_key) {
-            SNMP_ZERO(old_ref->usr_priv_key, old_ref->usr_priv_key_length);
-            SNMP_FREE(old_ref->usr_priv_key);
+        if (__v_0002->usr_priv_key_length && __v_0002->usr_priv_key) {
+            SNMP_ZERO(__v_0002->usr_priv_key, __v_0002->usr_priv_key_length);
+            SNMP_FREE(__v_0002->usr_priv_key);
         }
 
-        SNMP_ZERO(old_ref, sizeof(*old_ref));
-        SNMP_FREE(old_ref);
+        SNMP_ZERO(__v_0002, sizeof(*__v_0002));
+        SNMP_FREE(__v_0002);
 
     }
```

## diversevul / identifier / diversevul_test_002776

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,11 +1,11 @@
-QPDFTokenizer::findEI(PointerHolder<InputSource> input)
+QPDFTokenizer::findEI(PointerHolder<InputSource> __v_0010)
 {
-    if (! input.getPointer())
+    if (! __v_0010.getPointer())
     {
         return;
     }
 
-    qpdf_offset_t last_offset = input->getLastOffset();
-    qpdf_offset_t pos = input->tell();
+    qpdf_offset_t __v_0012 = __v_0010->getLastOffset();
+    qpdf_offset_t __v_0014 = __v_0010->tell();
 
     // Use QPDFWordTokenFinder to find EI surrounded by delimiters.
@@ -18,17 +18,17 @@
     // inline image, falling back to the old method if needed.
 
-    bool okay = false;
-    bool first_try = true;
-    while (! okay)
+    bool __v_0013 = false;
+    bool __v_0004 = true;
+    while (! __v_0013)
     {
-        QPDFWordTokenFinder f(input, "EI");
-        if (! input->findFirst("EI", input->tell(), 0, f))
+        QPDFWordTokenFinder __v_0003(__v_0010, "EI");
+        if (! __v_0010->findFirst("EI", __v_0010->tell(), 0, __v_0003))
         {
             break;
         }
-        this->m->inline_image_bytes = input->tell() - pos - 2;
+        this->m->inline_image_bytes = __v_0010->tell() - __v_0014 - 2;
 
-        QPDFTokenizer check;
-        bool found_bad = false;
+        QPDFTokenizer __v_0002;
+        bool __v_0006 = false;
         // Look at the next 10 tokens or up to EOF. The next inline
         // image's image data would look like bad tokens, but there
@@ -38,18 +38,18 @@
         // a BI and ID. If we get 10 good tokens in a row or hit EOF,
         // we can be pretty sure we've found the actual EI.
-        for (int i = 0; i < 10; ++i)
+        for (int __v_0009 = 0; __v_0009 < 10; ++__v_0009)
         {
-            QPDFTokenizer::Token t =
-                check.readToken(input, "checker", true);
-            token_type_e type = t.getType();
-            if (type == tt_eof)
+            QPDFTokenizer::Token __v_0015 =
+                __v_0002.readToken(__v_0010, "checker", true);
+            token_type_e __v_0016 = __v_0015.getType();
+            if (__v_0016 == tt_eof)
             {
-                okay = true;
+                __v_0013 = true;
             }
-            else if (type == tt_bad)
+            else if (__v_0016 == tt_bad)
             {
-                found_bad = true;
+                __v_0006 = true;
             }
-            else if (type == tt_word)
+            else if (__v_0016 == tt_word)
             {
                 // The qpdf tokenizer lumps alphabetic and otherwise
@@ -61,56 +61,56 @@
                 // which was already good enough for the vast majority
                 // of files.
-                bool found_alpha = false;
-                bool found_non_printable = false;
-                bool found_other = false;
-                std::string value = t.getValue();
-                for (std::string::iterator iter = value.begin();
-                     iter != value.end(); ++iter)
+                bool __v_0005 = false;
+                bool __v_0007 = false;
+                bool __v_0008 = false;
+                std::string __v_0017 = __v_0015.getValue();
+                for (std::string::iterator __v_0011 = __v_0017.begin();
+                     __v_0011 != __v_0017.end(); ++__v_0011)
                 {
-                    char ch = *iter;
-                    if (((ch >= 'a') && (ch <= 'z')) ||
-                        ((ch >= 'A') && (ch <= 'Z')) ||
-                        (ch == '*'))
+                    char __v_0001 = *__v_0011;
+                    if (((__v_0001 >= 'a') && (__v_0001 <= 'z')) ||
+                        ((__v_0001 >= 'A') && (__v_0001 <= 'Z')) ||
+                        (__v_0001 == '*'))
                     {
                         // Treat '*' as alpha since there are valid
                         // PDF operators that contain * along with
                         // alphabetic characters.
-                        found_alpha = true;
+                        __v_0005 = true;
                     }
-                    else if (((ch < 32) && (! isSpace(ch))) || (ch > 127))
+                    else if (((__v_0001 < 32) && (! isSpace(__v_0001))) || (__v_0001 > 127))
                     {
-                        found_non_printable = true;
+                        __v_0007 = true;
                         break;
                     }
                     else
                     {
-                        found_other = true;
+                        __v_0008 = true;
                     }
                 }
-                if (found_non_printable || (found_alpha && found_other))
+                if (__v_0007 || (__v_0005 && __v_0008))
                 {
-                    found_bad = true;
+                    __v_0006 = true;
                 }
             }
-            if (okay || found_bad)
+            if (__v_0013 || __v_0006)
             {
                 break;
             }
         }
-        if (! found_bad)
+        if (! __v_0006)
         {
-            okay = true;
+            __v_0013 = true;
         }
-        if (! okay)
+        if (! __v_0013)
         {
-            first_try = false;
+            __v_0004 = false;
         }
     }
-    if (okay && (! first_try))
+    if (__v_0013 && (! __v_0004))
     {
         QTC::TC("qpdf", "QPDFTokenizer found EI after more than one try");
     }
 
-    input->seek(pos, SEEK_SET);
-    input->setLastOffset(last_offset);
+    __v_0010->seek(__v_0014, SEEK_SET);
+    __v_0010->setLastOffset(__v_0012);
 }
```

## diversevul / identifier / diversevul_test_003143

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,4 +1,4 @@
-int mp_exch (mp_int * a, mp_int * b)
+int mp_exch (mp_int * __v_0001, mp_int * __v_0002)
 {
-    return fp_exch(a, b);
+    return fp_exch(__v_0001, __v_0002);
 }
```

## diversevul / identifier / diversevul_test_003245

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,18 +1,18 @@
-static int send_xmessage_using_XAUTHORITY(struct proclistlist *pll,int lokke,char *message){
+static int send_xmessage_using_XAUTHORITY(struct proclistlist *__v_0003,int __v_0001,char *__v_0002){
 
-  if(lokke==pll->length)
+  if(__v_0001==__v_0003->length)
     return 0;
 
   {
-    char *xa_filename=get_pid_environ_val(pll->proclist[lokke].pid,"XAUTHORITY");
-    if(xa_filename!=NULL){
-      if(send_xmessage(xa_filename,message)==1){
-	free(xa_filename);
+    char *__v_0004=get_pid_environ_val(__v_0003->proclist[__v_0001].pid,"XAUTHORITY");
+    if(__v_0004!=NULL){
+      if(send_xmessage(__v_0004,__v_0002)==1){
+	free(__v_0004);
 	return 1;
       }	
     }
-    free(xa_filename);
+    free(__v_0004);
   }
 
-  return send_xmessage_using_XAUTHORITY(pll,lokke+1,message);
+  return send_xmessage_using_XAUTHORITY(__v_0003,__v_0001+1,__v_0002);
 }
```

## diversevul / identifier / diversevul_test_003499

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,14 +1,14 @@
-DocumentSourceLookUp::DocumentSourceLookUp(NamespaceString fromNs,
-                                           std::string as,
-                                           const boost::intrusive_ptr<ExpressionContext>& expCtx)
-    : DocumentSource(kStageName, expCtx),
-      _fromNs(std::move(fromNs)),
-      _as(std::move(as)),
-      _variables(expCtx->variables),
-      _variablesParseState(expCtx->variablesParseState.copyWith(_variables.useIdGenerator())) {
-    const auto& resolvedNamespace = expCtx->getResolvedNamespace(_fromNs);
-    _resolvedNs = resolvedNamespace.ns;
-    _resolvedPipeline = resolvedNamespace.pipeline;
-    _fromExpCtx = expCtx->copyWith(_resolvedNs);
+DocumentSourceLookUp::DocumentSourceLookUp(NamespaceString __v_0003,
+                                           std::string __v_0001,
+                                           const boost::intrusive_ptr<ExpressionContext>& __v_0002)
+    : DocumentSource(kStageName, __v_0002),
+      _fromNs(std::move(__v_0003)),
+      _as(std::move(__v_0001)),
+      _variables(__v_0002->variables),
+      _variablesParseState(__v_0002->variablesParseState.copyWith(_variables.useIdGenerator())) {
+    const auto& __v_0004 = __v_0002->getResolvedNamespace(_fromNs);
+    _resolvedNs = __v_0004.ns;
+    _resolvedPipeline = __v_0004.pipeline;
+    _fromExpCtx = __v_0002->copyWith(_resolvedNs);
 
     _fromExpCtx->subPipelineDepth += 1;
```

## diversevul / deadcode / diversevul_test_000669

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -2,11 +2,18 @@
 					      struct sk_buff *skb)
 {
+	{ int __dc_0001 = 0; if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
 	struct ip_options *opt = &(IPCB(skb)->opt);
+	{ int __dc_0002 = 0; if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
 	struct ip_options *dopt = NULL;
 
+	{ int __dm_0003 = 0; for (int __fl_0004 = 0; __fl_0004 < 0; __fl_0004++) { __dm_0003 += __fl_0004; } }
 	if (opt && opt->optlen) {
+		{ int __dc_0005 = 0; if (__dc_0005 != 0) { __dc_0005 = __dc_0005 + 1; } }
 		int opt_size = optlength(opt);
+		{ int __dc_0006 = 0; if (__dc_0006 != 0) { __dc_0006 = __dc_0006 + 1; } }
 		dopt = kmalloc(opt_size, GFP_ATOMIC);
+		{ int __dm_0007 = 0; for (int __fl_0008 = 0; __fl_0008 < 0; __fl_0008++) { __dm_0007 += __fl_0008; } }
 		if (dopt) {
+			{ int __dc_0009 = 0; if (__dc_0009 != 0) { __dc_0009 = __dc_0009 + 1; } }
 			if (ip_options_echo(dopt, skb)) {
 				kfree(dopt);
```

## diversevul / deadcode / diversevul_test_001334

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -2,4 +2,5 @@
                u_int atype, const u_char *pptr, u_int len)
 {
+	{ int __dc_0001 = 0; if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
 	int i;
 	uint16_t af;
@@ -67,4 +68,5 @@
                         }
 			ND_TCHECK(tptr[0]);
+                        { int __dc_0002 = 0; if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
                         ND_PRINT((ndo, "%s", tok2str(bgp_as_path_segment_close_values,
 						"?", tptr[0])));
@@ -166,4 +168,5 @@
 		if (len % 4) {
 			ND_PRINT((ndo, "invalid len"));
+			{ int __dm_0003 = 0; for (int __fl_0004 = 0; __fl_0004 < 0; __fl_0004++) { __dm_0003 += __fl_0004; } }
 			break;
 		}
@@ -280,4 +283,5 @@
                             if (tlen < (int)sizeof(struct in6_addr)) {
                                 ND_PRINT((ndo, "invalid len"));
+                                { int __dc_0005 = 0; if (__dc_0005 != 0) { __dc_0005 = __dc_0005 + 1; } }
                                 tlen = 0;
                             } else {
@@ -367,4 +371,5 @@
 
 		if (snpa) {
+			{ int __dc_0006 = 0; if (__dc_0006 != 0) { __dc_0006 = __dc_0006 + 1; } }
 			ND_PRINT((ndo, "\n\t    %u SNPA", snpa));
 			for (/*nothing*/; snpa > 0; snpa--) {
@@ -775,4 +780,5 @@
 
                 tunnel_type = *(tptr+1);
+                { int __dm_0007 = 0; for (int __fl_0008 = 0; __fl_0008 < 0; __fl_0008++) { __dm_0007 += __fl_0008; } }
                 flags = *tptr;
                 tlen = len;
@@ -882,4 +888,5 @@
                     u_int aflags, alenlen, alen;
 
+                    { int __dc_0009 = 0; if (__dc_0009 != 0) { __dc_0009 = __dc_0009 + 1; } }
                     ND_TCHECK2(tptr[0], 2);
                     if (len < 2)
```

## diversevul / deadcode / diversevul_test_001355

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -1,22 +1,29 @@
 static NTSTATUS TCReadWriteDevice (BOOL write, PDEVICE_OBJECT deviceObject, PVOID buffer, LARGE_INTEGER offset, ULONG length)
 {
+	{ int __dc_0001 = 0; if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
 	NTSTATUS status;
 	IO_STATUS_BLOCK ioStatusBlock;
+	{ int __dc_0002 = 0; if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
 	PIRP irp;
 	KEVENT completionEvent;
 
+	{ int __dm_0003 = 0; for (int __fl_0004 = 0; __fl_0004 < 0; __fl_0004++) { __dm_0003 += __fl_0004; } }
 	ASSERT (KeGetCurrentIrql() <= APC_LEVEL);
 
 	KeInitializeEvent (&completionEvent, NotificationEvent, FALSE);
+	{ int __dc_0005 = 0; if (__dc_0005 != 0) { __dc_0005 = __dc_0005 + 1; } }
 	irp = IoBuildSynchronousFsdRequest (write ? IRP_MJ_WRITE : IRP_MJ_READ, deviceObject, buffer, length, &offset, &completionEvent, &ioStatusBlock);
 	if (!irp)
 		return STATUS_INSUFFICIENT_RESOURCES;
 
+	{ int __dc_0006 = 0; if (__dc_0006 != 0) { __dc_0006 = __dc_0006 + 1; } }
 	ObReferenceObject (deviceObject);
 	status = IoCallDriver (deviceObject, irp);
 
+	{ int __dm_0007 = 0; for (int __fl_0008 = 0; __fl_0008 < 0; __fl_0008++) { __dm_0007 += __fl_0008; } }
 	if (status == STATUS_PENDING)
 	{
 		status = KeWaitForSingleObject (&completionEvent, Executive, KernelMode, FALSE, NULL);
+		{ int __dc_0009 = 0; if (__dc_0009 != 0) { __dc_0009 = __dc_0009 + 1; } }
 		if (NT_SUCCESS (status))
 			status = ioStatusBlock.Status;
```

## diversevul / deadcode / diversevul_test_001435

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -9,6 +9,9 @@
       divide the max_length with mbmaxlen).
     */
+    { int __dc_0001 = 0; if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
     collation.set(cs, dv, metadata.repertoire());
+    { int __dc_0002 = 0; if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
     fix_char_length(metadata.char_length());
+    { int __dm_0003 = 0; for (int __fl_0004 = 0; __fl_0004 < 0; __fl_0004++) { __dm_0003 += __fl_0004; } }
     decimals= NOT_FIXED_DEC;
   }
```

## diversevul / deadcode / diversevul_test_002126

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -1,13 +1,18 @@
 void DocumentSourceGroup::freeMemory() {
+    { int __dc_0001 = 0; if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
     invariant(_groups);
     for (auto&& group : *_groups) {
+        { int __dc_0002 = 0; if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
         for (size_t i = 0; i < group.second.size(); i++) {
             // Subtract the current usage.
+            { int __dm_0003 = 0; for (int __fl_0004 = 0; __fl_0004 < 0; __fl_0004++) { __dm_0003 += __fl_0004; } }
             _memoryTracker.update(_accumulatedFields[i].fieldName,
                                   -1 * group.second[i]->getMemUsage());
 
+            { int __dc_0005 = 0; if (__dc_0005 != 0) { __dc_0005 = __dc_0005 + 1; } }
             group.second[i]->reduceMemoryConsumptionIfAble();
 
             // Update the memory usage for this AccumulationStatement.
+            { int __dc_0006 = 0; if (__dc_0006 != 0) { __dc_0006 = __dc_0006 + 1; } }
             _memoryTracker.update(_accumulatedFields[i].fieldName, group.second[i]->getMemUsage());
         }
```

## diversevul / deadcode / diversevul_test_002509

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -1,7 +1,10 @@
 _g_locale_get_charset_aliases (void)
 {
+  { int __dc_0001 = 0; if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
   const char *cp;
 
+  { int __dc_0002 = 0; if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
   cp = charset_aliases;
+  { int __dm_0003 = 0; for (int __fl_0004 = 0; __fl_0004 < 0; __fl_0004++) { __dm_0003 += __fl_0004; } }
   if (cp == NULL)
     {
@@ -163,7 +166,9 @@
 #endif
 
+      { int __dc_0005 = 0; if (__dc_0005 != 0) { __dc_0005 = __dc_0005 + 1; } }
       charset_aliases = cp;
     }
 
+  { int __dc_0006 = 0; if (__dc_0006 != 0) { __dc_0006 = __dc_0006 + 1; } }
   return cp;
 }
```

## diversevul / deadcode / diversevul_test_002689

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -1,7 +1,9 @@
 struct tcp_sock_t *tcp_open(uint16_t port)
 {
+	{ int __dc_0001 = (port ^ port); if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
 	struct tcp_sock_t *this = calloc(1, sizeof *this);
 	if (this == NULL) {
 		ERR("callocing this failed");
+		{ int __dc_0002 = (port ^ port); if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
 		goto error;
 	}
@@ -10,4 +12,5 @@
 	this->sd = -1;
 	this->sd = socket(AF_INET6, SOCK_STREAM, 0);
+	{ int __dm_0003 = 0; for (int __fl_0004 = 0; __fl_0004 < 0; __fl_0004++) { __dm_0003 += __fl_0004; } }
 	if (this->sd < 0) {
 		ERR("sockect open failed");
@@ -16,7 +19,9 @@
 
 	// Configure socket params
+	{ int __dc_0005 = (port ^ port); if (__dc_0005 != 0) { __dc_0005 = __dc_0005 + 1; } }
 	struct sockaddr_in6 addr;
 	memset(&addr, 0, sizeof addr);
 	addr.sin6_family = AF_INET6;
+	{ int __dc_0006 = (port ^ port); if (__dc_0006 != 0) { __dc_0006 = __dc_0006 + 1; } }
 	addr.sin6_port = htons(port);
 	addr.sin6_addr = in6addr_any;
@@ -26,4 +31,5 @@
 	        (struct sockaddr *)&addr,
 	        sizeof addr) < 0) {
+		{ int __dm_0007 = 0; for (int __fl_0008 = 0; __fl_0008 < 0; __fl_0008++) { __dm_0007 += __fl_0008; } }
 		if (g_options.only_desired_port == 1)
 			ERR("Bind on port failed. "
@@ -34,4 +40,5 @@
 	// Let kernel over-accept max number of connections
 	if (listen(this->sd, HTTP_MAX_PENDING_CONNS) < 0) {
+		{ int __dc_0009 = (port ^ port); if (__dc_0009 != 0) { __dc_0009 = __dc_0009 + 1; } }
 		ERR("listen failed on socket");
 		goto error;
```

## diversevul / deadcode / diversevul_test_002999

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -2,19 +2,26 @@
 			   struct mm_walk *walk)
 {
+	{ int __dc_0001 = (addr ^ addr); if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
 	struct mem_size_stats *mss = walk->private;
 	struct vm_area_struct *vma = mss->vma;
+	{ int __dc_0002 = (addr ^ addr); if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
 	pte_t *pte;
 	spinlock_t *ptl;
 
+	{ int __dm_0003 = 0; for (int __fl_0004 = 0; __fl_0004 < 0; __fl_0004++) { __dm_0003 += __fl_0004; } }
 	spin_lock(&walk->mm->page_table_lock);
 	if (pmd_trans_huge(*pmd)) {
+		{ int __dc_0005 = (addr ^ addr); if (__dc_0005 != 0) { __dc_0005 = __dc_0005 + 1; } }
 		if (pmd_trans_splitting(*pmd)) {
 			spin_unlock(&walk->mm->page_table_lock);
+			{ int __dc_0006 = (addr ^ addr); if (__dc_0006 != 0) { __dc_0006 = __dc_0006 + 1; } }
 			wait_split_huge_page(vma->anon_vma, pmd);
 		} else {
 			smaps_pte_entry(*(pte_t *)pmd, addr,
 					HPAGE_PMD_SIZE, walk);
+			{ int __dm_0007 = 0; for (int __fl_0008 = 0; __fl_0008 < 0; __fl_0008++) { __dm_0007 += __fl_0008; } }
 			spin_unlock(&walk->mm->page_table_lock);
 			mss->anonymous_thp += HPAGE_PMD_SIZE;
+			{ int __dc_0009 = (addr ^ addr); if (__dc_0009 != 0) { __dc_0009 = __dc_0009 + 1; } }
 			return 0;
 		}
```

## diversevul / deadcode / diversevul_test_003275

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -1,11 +1,18 @@
 TEST(IndexBoundsBuilderTest, ExistsFalseWithMockCollatorIsInexactFetch) {
+    { int __dc_0001 = 0; if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
     CollatorInterfaceMock collator(CollatorInterfaceMock::MockType::kReverseString);
+    { int __dc_0002 = 0; if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
     auto testIndex = buildSimpleIndexEntry();
+    { int __dm_0003 = 0; for (int __fl_0004 = 0; __fl_0004 < 0; __fl_0004++) { __dm_0003 += __fl_0004; } }
     testIndex.collator = &collator;
 
+    { int __dc_0005 = 0; if (__dc_0005 != 0) { __dc_0005 = __dc_0005 + 1; } }
     BSONObj obj = fromjson("{a: {$exists: false}}");
+    { int __dc_0006 = 0; if (__dc_0006 != 0) { __dc_0006 = __dc_0006 + 1; } }
     auto expr = parseMatchExpression(obj);
+    { int __dm_0007 = 0; for (int __fl_0008 = 0; __fl_0008 < 0; __fl_0008++) { __dm_0007 += __fl_0008; } }
     BSONElement elt = obj.firstElement();
 
+    { int __dc_0009 = 0; if (__dc_0009 != 0) { __dc_0009 = __dc_0009 + 1; } }
     OrderedIntervalList oil;
     IndexBoundsBuilder::BoundsTightness tightness;
```

## diversevul / deadcode / diversevul_test_003332

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -2,6 +2,8 @@
                       int correct_attrs)
 {
+    { int __dc_0001 = (copy_ns ^ copy_ns); if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
     const struct lyxml_ns *tmp_ns;
     struct lyxml_elem *elem_root, *ns_root, *tmp, *iter;
+    { int __dc_0002 = (copy_ns ^ copy_ns); if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
     struct lyxml_attr *attr;
 
@@ -10,4 +12,5 @@
 
     LY_TREE_DFS_BEGIN(elem, tmp, iter) {
+        { int __dm_0003 = 0; for (int __fl_0004 = 0; __fl_0004 < 0; __fl_0004++) { __dm_0003 += __fl_0004; } }
         if (iter->ns) {
             /* find the root of elem NS */
@@ -15,6 +18,8 @@
 
             /* elem NS is defined outside elem subtree */
+            { int __dc_0005 = (copy_ns ^ copy_ns); if (__dc_0005 != 0) { __dc_0005 = __dc_0005 + 1; } }
             if (ns_root != elem_root) {
                 if (copy_ns) {
+                    { int __dc_0006 = (copy_ns ^ copy_ns); if (__dc_0006 != 0) { __dc_0006 = __dc_0006 + 1; } }
                     tmp_ns = iter->ns;
                     /* we may have already copied the NS over? */
@@ -22,8 +27,10 @@
 
                     /* we haven't copied it over, copy it now */
+                    { int __dm_0007 = 0; for (int __fl_0008 = 0; __fl_0008 < 0; __fl_0008++) { __dm_0007 += __fl_0008; } }
                     if (!iter->ns) {
                         iter->ns = (struct lyxml_ns *)lyxml_dup_attr(ctx, iter, (struct lyxml_attr *)tmp_ns);
                     }
                 } else {
+                    { int __dc_0009 = (copy_ns ^ copy_ns); if (__dc_0009 != 0) { __dc_0009 = __dc_0009 + 1; } }
                     iter->ns = NULL;
                 }
```

## diversevul / controlflow / diversevul_test_000107

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -9,28 +9,33 @@
 	if (IS_ERR(ep))
 		return PTR_ERR(ep);
-	mutex_lock(&rp->printf_lock);
-	ptr.cnt = 0;
-	ptr.pbuf = rp->printf_buf;
-	ptr.limit = rp->printf_size;
+	{ int __st_0002 = 0;
+	while (__st_0002 < 4) {
+	    if (__st_0002 == 0) { mutex_lock(&rp->printf_lock); __st_0002 = 1; }
+	    else if (__st_0002 == 1) { ptr.cnt = 0; __st_0002 = 2; }
+	    else if (__st_0002 == 2) { ptr.pbuf = rp->printf_buf; __st_0002 = 3; }
+	    else if (__st_0002 == 3) { ptr.limit = rp->printf_size; __st_0002 = 4; }
+	} }
 
 	mon_text_read_head_u(rp, &ptr, ep);
 	if (ep->type == 'E') {
-		mon_text_read_statset(rp, &ptr, ep);
+		{ if (((nbytes | nbytes) == nbytes)) { mon_text_read_statset(rp, &ptr, ep); } }
 	} else if (ep->xfertype == USB_ENDPOINT_XFER_ISOC) {
 		mon_text_read_isostat(rp, &ptr, ep);
 		mon_text_read_isodesc(rp, &ptr, ep);
-	} else if (ep->xfertype == USB_ENDPOINT_XFER_INT) {
+	} else { int __p_0001 = (ep->xfertype == USB_ENDPOINT_XFER_INT) ? 1 : 0;
+    if (__p_0001) {
 		mon_text_read_intstat(rp, &ptr, ep);
-	} else {
+	}
+    if (!__p_0001) {
 		mon_text_read_statset(rp, &ptr, ep);
-	}
+	} }
 	ptr.cnt += snprintf(ptr.pbuf + ptr.cnt, ptr.limit - ptr.cnt,
 	    " %d", ep->length);
-	mon_text_read_data(rp, &ptr, ep);
+	{ if (((nbytes | nbytes) == nbytes)) { mon_text_read_data(rp, &ptr, ep); } }
 
 	if (copy_to_user(buf, rp->printf_buf, ptr.cnt))
 		ptr.cnt = -EFAULT;
 	mutex_unlock(&rp->printf_lock);
-	kmem_cache_free(rp->e_slab, ep);
+	{ if (((nbytes | nbytes) == nbytes)) { kmem_cache_free(rp->e_slab, ep); } }
 	return ptr.cnt;
 }
```

## diversevul / controlflow / diversevul_test_000390

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -17,8 +17,8 @@
 		st = tell = php_stream_tell(pass->filefp);
 		/* copy the local files, central directory, and the zip comment to generate the hash */
-		php_stream_seek(pass->filefp, 0, SEEK_SET);
+		{ if ((1)) { php_stream_seek(pass->filefp, 0, SEEK_SET); } }
 		php_stream_copy_to_stream_ex(pass->filefp, newfile, tell, NULL);
 		tell = php_stream_tell(pass->centralfp);
-		php_stream_seek(pass->centralfp, 0, SEEK_SET);
+		{ if ((1)) { php_stream_seek(pass->centralfp, 0, SEEK_SET); } }
 		php_stream_copy_to_stream_ex(pass->centralfp, newfile, tell, NULL);
 		if (metadata->s) {
@@ -29,5 +29,5 @@
 			if (pass->error) {
 				char *save = *(pass->error);
-				spprintf(pass->error, 0, "phar error: unable to write signature to zip-based phar: %s", save);
+				{ if ((1)) { spprintf(pass->error, 0, "phar error: unable to write signature to zip-based phar: %s", save); } }
 				efree(save);
 			}
@@ -37,8 +37,11 @@
 		}
 
-		entry.filename = ".phar/signature.bin";
-		entry.filename_len = sizeof(".phar/signature.bin")-1;
-		entry.fp = php_stream_fopen_tmpfile();
-		entry.fp_type = PHAR_MOD;
+		{ int __st_0005 = 0;
+		while (__st_0005 < 4) {
+		    if (__st_0005 == 0) { entry.filename = ".phar/signature.bin"; __st_0005 = 1; }
+		    else if (__st_0005 == 1) { entry.filename_len = sizeof(".phar/signature.bin")-1; __st_0005 = 2; }
+		    else if (__st_0005 == 2) { entry.fp = php_stream_fopen_tmpfile(); __st_0005 = 3; }
+		    else if (__st_0005 == 3) { entry.fp_type = PHAR_MOD; __st_0005 = 4; }
+		} }
 		entry.is_modified = 1;
 		if (entry.fp == NULL) {
```

## diversevul / controlflow / diversevul_test_001549

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -7,5 +7,5 @@
 	iorq = g_iorequest;
 	prev = NULL;
-	while (iorq != NULL)
+	while ((iorq != NULL) && (1 == 1))
 	{
 		/* Only remove from table when major is not set, or when correct major is supplied.
@@ -14,6 +14,6 @@
 		{
 			result = 0;
-			rdpdr_send_completion(iorq->device, iorq->id, status, result, (uint8 *) "",
-					      1);
+			{ if ((1)) { rdpdr_send_completion(iorq->device, iorq->id, status, result, (uint8 *) "",
+					      1); } }
 
 			iorq = rdpdr_remove_iorequest(prev, iorq);
```

## diversevul / controlflow / diversevul_test_001571

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -2,4 +2,5 @@
                    const struct pcap_pkthdr *h, register const u_char *p)
 {
+        { int __p_0001 = (0 == 0); if (__p_0001 && (__p_0001 == 1)) { __p_0001 = __p_0001; } }
         int llc_hdrlen;
 
```

## diversevul / controlflow / diversevul_test_001698

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -2,7 +2,7 @@
         : Tool( name , access , "" , "" , false ) , _objcheck( objcheck ) {
 
-        add_options()
+        { if ((1)) { add_options()
         ("objcheck" , "validate object before inserting" )
         ("filter" , po::value<string>() , "filter to apply before inserting" )
-        ;
+        ; } }
     }
```

## diversevul / controlflow / diversevul_test_001796

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -3,9 +3,11 @@
 	gboolean nb_mode;
 
-	nb_mode = sock_is_nonblocking_mode(session->sock);
-
-	session->sock->ssl_cert_auto_accept = session->ssl_cert_auto_accept;
-	session->sock->gnutls_priority = session->gnutls_priority;
-	session->sock->use_tls_sni = session->use_tls_sni;
+	{ int __st_0001 = 0;
+	while (__st_0001 < 4) {
+	    if (__st_0001 == 0) { nb_mode = sock_is_nonblocking_mode(session->sock); __st_0001 = 1; }
+	    else if (__st_0001 == 1) { session->sock->ssl_cert_auto_accept = session->ssl_cert_auto_accept; __st_0001 = 2; }
+	    else if (__st_0001 == 2) { session->sock->gnutls_priority = session->gnutls_priority; __st_0001 = 3; }
+	    else if (__st_0001 == 3) { session->sock->use_tls_sni = session->use_tls_sni; __st_0001 = 4; }
+	} }
 
 	if (nb_mode)
@@ -15,5 +17,5 @@
 		g_warning("couldn't start STARTTLS session.");
 		if (nb_mode)
-			sock_set_nonblocking_mode(session->sock, session->nonblocking);
+			{ if ((1)) { sock_set_nonblocking_mode(session->sock, session->nonblocking); } }
 		return -1;
 	}
```

## diversevul / controlflow / diversevul_test_001959

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -1,7 +1,7 @@
     Http::Status dispatch(const Buffer::Instance& data, ConnectionImpl& connection) {
       Http::Status status = Http::okStatus();
-      buffer_.add(data);
+      { if ((1)) { buffer_.add(data); } }
       if (!dispatching_) {
-        while (buffer_.length() > 0) {
+        while ((buffer_.length() > 0) && (1 == 1)) {
           dispatching_ = true;
           status = connection.dispatch(buffer_);
```

## diversevul / controlflow / diversevul_test_002485

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -8,6 +8,6 @@
 
   std::vector<HeaderUtility::HeaderDataPtr> header_data;
-  header_data.push_back(
-      std::make_unique<HeaderUtility::HeaderData>(parseHeaderMatcherFromYaml(yaml)));
+  { if ((1)) { header_data.push_back(
+      std::make_unique<HeaderUtility::HeaderData>(parseHeaderMatcherFromYaml(yaml))); } }
   EXPECT_FALSE(HeaderUtility::matchHeaders(headers, header_data));
 
```

## diversevul / controlflow / diversevul_test_003012

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -2,4 +2,5 @@
 				      int mode, unsigned int dev)
 {
+	{ int __p_0001 = (0 == 0); if (__p_0001 && (__p_0001 == 1)) { __p_0001 = __p_0001; } }
 	return 0;
 }
```

## diversevul / controlflow / diversevul_test_003590

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -2,6 +2,6 @@
                            GPtrArray    *other_array)
 {
-  flatpak_bwrap_append_argsv (bwrap,
+  { if ((1)) { flatpak_bwrap_append_argsv (bwrap,
                               (char **) other_array->pdata,
-                              other_array->len);
+                              other_array->len); } }
 }
```

## bigvul / identifier / bigvul_test_002200

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,5 +1,5 @@
-XML_FreeContentModel(XML_Parser parser, XML_Content *model)
+XML_FreeContentModel(XML_Parser __v_0002, XML_Content *__v_0001)
 {
-  if (parser != NULL)
-    FREE(parser, model);
+  if (__v_0002 != NULL)
+    FREE(__v_0002, __v_0001);
 }
```

## bigvul / identifier / bigvul_test_008005

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,8 +1,8 @@
-static void pdf_run_Tf(fz_context *ctx, pdf_processor *proc, const char *name, pdf_font_desc *font, float size)
+static void pdf_run_Tf(fz_context *__v_0001, pdf_processor *__v_0006, const char *__v_0004, pdf_font_desc *__v_0002, float __v_0007)
 {
-	pdf_run_processor *pr = (pdf_run_processor *)proc;
-	pdf_gstate *gstate = pr->gstate + pr->gtop;
-	pdf_drop_font(ctx, gstate->text.font);
-	gstate->text.font = pdf_keep_font(ctx, font);
-	gstate->text.size = size;
+	pdf_run_processor *__v_0005 = (pdf_run_processor *)__v_0006;
+	pdf_gstate *__v_0003 = __v_0005->gstate + __v_0005->gtop;
+	pdf_drop_font(__v_0001, __v_0003->text.font);
+	__v_0003->text.font = pdf_keep_font(__v_0001, __v_0002);
+	__v_0003->text.size = __v_0007;
 }
```

## bigvul / identifier / bigvul_test_009400

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,28 +1,28 @@
-XineramaXvStopVideo(ClientPtr client)
+XineramaXvStopVideo(ClientPtr __v_0001)
 {
-    int result, i;
-    PanoramiXRes *draw, *port;
+    int __v_0004, i;
+    PanoramiXRes *__v_0002, *__v_0003;
 
     REQUEST(xvStopVideoReq);
     REQUEST_SIZE_MATCH(xvStopVideoReq);
 
-    result = dixLookupResourceByClass((void **) &draw, stuff->drawable,
-                                      XRC_DRAWABLE, client, DixWriteAccess);
-    if (result != Success)
-        return (result == BadValue) ? BadDrawable : result;
+    __v_0004 = dixLookupResourceByClass((void **) &__v_0002, stuff->drawable,
+                                      XRC_DRAWABLE, __v_0001, DixWriteAccess);
+    if (__v_0004 != Success)
+        return (__v_0004 == BadValue) ? BadDrawable : __v_0004;
 
-    result = dixLookupResourceByType((void **) &port, stuff->port,
-                                     XvXRTPort, client, DixReadAccess);
-    if (result != Success)
-        return result;
+    __v_0004 = dixLookupResourceByType((void **) &__v_0003, stuff->port,
+                                     XvXRTPort, __v_0001, DixReadAccess);
+    if (__v_0004 != Success)
+        return __v_0004;
 
     FOR_NSCREENS_BACKWARD(i) {
-        if (port->info[i].id) {
-            stuff->drawable = draw->info[i].id;
-            stuff->port = port->info[i].id;
-            result = ProcXvStopVideo(client);
+        if (__v_0003->info[i].id) {
+            stuff->drawable = __v_0002->info[i].id;
+            stuff->port = __v_0003->info[i].id;
+            __v_0004 = ProcXvStopVideo(__v_0001);
         }
     }
 
-    return result;
+    return __v_0004;
 }
```

## bigvul / identifier / bigvul_test_009559

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,15 +1,15 @@
-static int dm_wait_for_completion(struct mapped_device *md, long task_state)
+static int dm_wait_for_completion(struct mapped_device *__v_0001, long __v_0003)
 {
-	int r = 0;
+	int __v_0002 = 0;
 	DEFINE_WAIT(wait);
 
 	while (1) {
-		prepare_to_wait(&md->wait, &wait, task_state);
+		prepare_to_wait(&__v_0001->wait, &wait, __v_0003);
 
-		if (!md_in_flight(md))
+		if (!md_in_flight(__v_0001))
 			break;
 
-		if (signal_pending_state(task_state, current)) {
-			r = -EINTR;
+		if (signal_pending_state(__v_0003, current)) {
+			__v_0002 = -EINTR;
 			break;
 		}
@@ -17,6 +17,6 @@
 		io_schedule();
 	}
-	finish_wait(&md->wait, &wait);
+	finish_wait(&__v_0001->wait, &wait);
 
-	return r;
+	return __v_0002;
 }
```

## bigvul / identifier / bigvul_test_010474

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,34 +1,34 @@
-dissect_u3v_read_mem_ack(proto_tree *u3v_telegram_tree, tvbuff_t *tvb, packet_info *pinfo, gint startoffset, gint length, u3v_conv_info_t *u3v_conv_info, gencp_transaction_t * gencp_trans)
+dissect_u3v_read_mem_ack(proto_tree *__v_0014, tvbuff_t *__v_0012, packet_info *__v_0010, gint __v_0011, gint __v_0008, u3v_conv_info_t *__v_0013, gencp_transaction_t * __v_0004)
 {
-    guint64 addr = 0;
-    const gchar *address_string = NULL;
-    gboolean is_custom_register = FALSE;
-    gboolean have_address = (0 != gencp_trans->cmd_frame);
-    proto_item *item = NULL;
-    guint offset = startoffset;
-    guint byte_count = (length);
+    guint64 __v_0001 = 0;
+    const gchar *__v_0002 = NULL;
+    gboolean __v_0006 = FALSE;
+    gboolean __v_0005 = (0 != __v_0004->cmd_frame);
+    proto_item *__v_0007 = NULL;
+    guint __v_0009 = __v_0011;
+    guint __v_0003 = (__v_0008);
 
-    addr = gencp_trans->address;
-    dissect_u3v_register_bases(addr, tvb, startoffset, u3v_conv_info);
-    if (have_address) {
-        address_string = get_register_name_from_address(addr, &is_custom_register, u3v_conv_info);
+    __v_0001 = __v_0004->address;
+    dissect_u3v_register_bases(__v_0001, __v_0012, __v_0011, __v_0013);
+    if (__v_0005) {
+        __v_0002 = get_register_name_from_address(__v_0001, &__v_0006, __v_0013);
         /* Fill in Wireshark GUI Info column */
-        col_append_fstr(pinfo->cinfo, COL_INFO, "%s", address_string);
+        col_append_fstr(__v_0010->cinfo, COL_INFO, "%s", __v_0002);
     }
 
 
     /* Subtree initialization for Payload Data: READMEM_ACK */
-    item = proto_tree_add_item(u3v_telegram_tree, hf_u3v_scd_ack_readmem_ack, tvb, startoffset, length, ENC_NA);
-    u3v_telegram_tree = proto_item_add_subtree(item, ett_u3v_payload_cmd);
+    __v_0007 = proto_tree_add_item(__v_0014, hf_u3v_scd_ack_readmem_ack, __v_0012, __v_0011, __v_0008, ENC_NA);
+    __v_0014 = proto_item_add_subtree(__v_0007, ett_u3v_payload_cmd);
 
     /* Bootstrap register known address */
-    if (have_address) {
-        item = proto_tree_add_uint64(u3v_telegram_tree, hf_u3v_address, tvb, 0,0 , addr);
-        PROTO_ITEM_SET_GENERATED(item);
+    if (__v_0005) {
+        __v_0007 = proto_tree_add_uint64(__v_0014, hf_u3v_address, __v_0012, 0,0 , __v_0001);
+        PROTO_ITEM_SET_GENERATED(__v_0007);
 
-        if (is_known_bootstrap_register(addr, u3v_conv_info)) {
-            dissect_u3v_register(addr, u3v_telegram_tree, tvb, offset, byte_count, u3v_conv_info);
+        if (is_known_bootstrap_register(__v_0001, __v_0013)) {
+            dissect_u3v_register(__v_0001, __v_0014, __v_0012, __v_0009, __v_0003, __v_0013);
         } else {
-            proto_tree_add_item(u3v_telegram_tree, hf_u3v_custom_memory_data, tvb, startoffset, length, ENC_NA);
+            proto_tree_add_item(__v_0014, hf_u3v_custom_memory_data, __v_0012, __v_0011, __v_0008, ENC_NA);
         }
     }
```

## bigvul / identifier / bigvul_test_011340

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,4 +1,4 @@
 int Init_IPFIX(void) {
-int i;
+int __v_0002;
 
 	cache.lookup_info	    = (struct element_param_s *)calloc(65536, sizeof(struct element_param_s));
@@ -9,10 +9,10 @@
 	}
 
-	for (i=1; ipfix_element_map[i].id != 0; i++ ) {
-		uint32_t Type = ipfix_element_map[i].id;
-		if ( cache.lookup_info[Type].index == 0 ) 
-			cache.lookup_info[Type].index  = i;
+	for (__v_0002=1; ipfix_element_map[__v_0002].id != 0; __v_0002++ ) {
+		uint32_t __v_0001 = ipfix_element_map[__v_0002].id;
+		if ( cache.lookup_info[__v_0001].index == 0 ) 
+			cache.lookup_info[__v_0001].index  = __v_0002;
 	}
-	cache.max_ipfix_elements = i;
+	cache.max_ipfix_elements = __v_0002;
 	cache.input_order = NULL;
 
```

## bigvul / identifier / bigvul_test_014169

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,21 +1,21 @@
-static void fbFetchSolid(PicturePtr pict, int x, int y, int width, CARD32 *buffer, CARD32 *mask, CARD32 maskBits)
+static void fbFetchSolid(PicturePtr __v_0010, int __v_0013, int __v_0015, int __v_0012, CARD32 *__v_0003, CARD32 *__v_0008, CARD32 __v_0009)
 {
-    FbBits *bits;
-    FbStride stride;
-    int bpp;
-    int xoff, yoff;
-    CARD32 color;
-    CARD32 *end;
-    fetchPixelProc fetch = fetchPixelProcForPicture(pict);
-    miIndexedPtr indexed = (miIndexedPtr) pict->pFormat->index.devPrivate;
+    FbBits *__v_0001;
+    FbStride __v_0011;
+    int __v_0002;
+    int __v_0014, __v_0016;
+    CARD32 __v_0004;
+    CARD32 *__v_0005;
+    fetchPixelProc __v_0006 = fetchPixelProcForPicture(__v_0010);
+    miIndexedPtr __v_0007 = (miIndexedPtr) __v_0010->pFormat->index.devPrivate;
 
-    fbGetDrawable (pict->pDrawable, bits, stride, bpp, xoff, yoff);
-    bits += yoff*stride + (xoff*bpp >> FB_SHIFT);
+    fbGetDrawable (__v_0010->pDrawable, __v_0001, __v_0011, __v_0002, __v_0014, __v_0016);
+    __v_0001 += __v_0016*__v_0011 + (__v_0014*__v_0002 >> FB_SHIFT);
 
-    color = fetch(bits, 0, indexed);
+    __v_0004 = __v_0006(__v_0001, 0, __v_0007);
 
-    end = buffer + width;
-    while (buffer < end)
-        WRITE(buffer++, color);
-    fbFinishAccess (pict->pDrawable);
+    __v_0005 = __v_0003 + __v_0012;
+    while (__v_0003 < __v_0005)
+        WRITE(__v_0003++, __v_0004);
+    fbFinishAccess (__v_0010->pDrawable);
 }
```

## bigvul / identifier / bigvul_test_015156

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,4 +1,4 @@
-static inline struct sched_entity *parent_entity(struct sched_entity *se)
+static inline struct sched_entity *parent_entity(struct sched_entity *__v_0001)
 {
-	return se->parent;
+	return __v_0001->parent;
 }
```

## bigvul / identifier / bigvul_test_016383

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,5 +1,5 @@
-void InspectorOverlay::setPausedInDebuggerMessage(const String* message)
+void InspectorOverlay::setPausedInDebuggerMessage(const String* __v_0001)
 {
-    m_pausedInDebuggerMessage = message ? *message : String();
+    m_pausedInDebuggerMessage = __v_0001 ? *__v_0001 : String();
     update();
 }
```

## bigvul / identifier / bigvul_test_016391

- [x] semantics preserved

```diff
--- original
+++ identifier
@@ -1,11 +1,11 @@
-inline void Splash::drawPixel(SplashPipe *pipe, int x, int y, GBool noClip) {
-  if (unlikely(y < 0))
+inline void Splash::drawPixel(SplashPipe *__v_0002, int __v_0003, int __v_0004, GBool __v_0001) {
+  if (unlikely(__v_0004 < 0))
     return;
 
-  if (noClip || state->clip->test(x, y)) {
-    pipeSetXY(pipe, x, y);
-    (this->*pipe->run)(pipe);
-    updateModX(x);
-    updateModY(y);
+  if (__v_0001 || state->clip->test(__v_0003, __v_0004)) {
+    pipeSetXY(__v_0002, __v_0003, __v_0004);
+    (this->*__v_0002->run)(__v_0002);
+    updateModX(__v_0003);
+    updateModY(__v_0004);
   }
 }
```

## bigvul / deadcode / bigvul_test_005312

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -1,15 +1,20 @@
 main (int argc, char *argv[]) {
     /* set up tests */
+    { int __dc_0001 = (argc ^ argc); if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
     g_type_init();
     g_test_init(&argc, &argv, NULL);
 
+    { int __dc_0002 = (argc ^ argc); if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
     g_test_add("/test-command/set-variable",   struct EventFixture, NULL, event_fixture_setup, test_set_variable, event_fixture_teardown);
     g_test_add("/test-command/event",          struct EventFixture, NULL, event_fixture_setup, test_event,        event_fixture_teardown);
 
+    { int __dm_0003 = 0; for (int __fl_0004 = 0; __fl_0004 < 0; __fl_0004++) { __dm_0003 += __fl_0004; } }
     g_test_add_func("/test-command/print",          test_print);
     g_test_add_func("/test-command/scroll",         test_scroll);
+    { int __dc_0005 = (argc ^ argc); if (__dc_0005 != 0) { __dc_0005 = __dc_0005 + 1; } }
     g_test_add_func("/test-command/toggle-status",  test_toggle_status);
     g_test_add_func("/test-command/sync-sh",        test_sync_sh);
 
+    { int __dc_0006 = (argc ^ argc); if (__dc_0006 != 0) { __dc_0006 = __dc_0006 + 1; } }
     g_test_add_func("/test-command/js",             test_js);
 
@@ -17,4 +22,5 @@
      * splitting into a separate file yet */
     g_test_add_func("/test-command/run_handler/arg-order",      test_run_handler_arg_order);
+    { int __dm_0007 = 0; for (int __fl_0008 = 0; __fl_0008 < 0; __fl_0008++) { __dm_0007 += __fl_0008; } }
     g_test_add_func("/test-command/run_handler/expand",         test_run_handler_expand);
 
@@ -22,4 +28,5 @@
     initialize(argc, argv);
 
+    { int __dc_0009 = (argc ^ argc); if (__dc_0009 != 0) { __dc_0009 = __dc_0009 + 1; } }
     uzbl.state.config_file = "/tmp/uzbl-config";
     uzbl.comm.fifo_path = "/tmp/some-nonexistant-fifo";
```

## bigvul / deadcode / bigvul_test_008187

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -2,10 +2,15 @@
     std::vector<uint8>* memory,
     scoped_ptr<skia::PlatformCanvas>* canvas) {
+  { int __dc_0001 = 0; if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
   const size_t size = BitmapSizeForPluginRect(plugin_rect_);
+  { int __dc_0002 = 0; if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
   memory->resize(size);
+  { int __dm_0003 = 0; for (int __fl_0004 = 0; __fl_0004 < 0; __fl_0004++) { __dm_0003 += __fl_0004; } }
   if (memory->size() != size)
     return false;
+  { int __dc_0005 = 0; if (__dc_0005 != 0) { __dc_0005 = __dc_0005 + 1; } }
   canvas->reset(new skia::PlatformCanvas(
       plugin_rect_.width(), plugin_rect_.height(), true, &((*memory)[0])));
+  { int __dc_0006 = 0; if (__dc_0006 != 0) { __dc_0006 = __dc_0006 + 1; } }
   return true;
 }
```

## bigvul / deadcode / bigvul_test_008644

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -1,13 +1,20 @@
 static int i8042_set_reset(const char *val, const struct kernel_param *kp)
 {
+	{ int __dc_0001 = 0; if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
 	enum i8042_controller_reset_mode *arg = kp->arg;
+	{ int __dc_0002 = 0; if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
 	int error;
+	{ int __dm_0003 = 0; for (int __fl_0004 = 0; __fl_0004 < 0; __fl_0004++) { __dm_0003 += __fl_0004; } }
 	bool reset;
 
+	{ int __dc_0005 = 0; if (__dc_0005 != 0) { __dc_0005 = __dc_0005 + 1; } }
 	if (val) {
+		{ int __dc_0006 = 0; if (__dc_0006 != 0) { __dc_0006 = __dc_0006 + 1; } }
 		error = kstrtobool(val, &reset);
+		{ int __dm_0007 = 0; for (int __fl_0008 = 0; __fl_0008 < 0; __fl_0008++) { __dm_0007 += __fl_0008; } }
 		if (error)
 			return error;
 	} else {
+		{ int __dc_0009 = 0; if (__dc_0009 != 0) { __dc_0009 = __dc_0009 + 1; } }
 		reset = true;
 	}
```

## bigvul / deadcode / bigvul_test_009902

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -1,4 +1,6 @@
  bool ClipboardUtil::HasUrl(IDataObject* data_object, bool convert_filenames) {
+   { int __dc_0001 = 0; if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
    DCHECK(data_object);
+   { int __dc_0002 = 0; if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
    return HasData(data_object, Clipboard::GetMozUrlFormatType()) ||
           HasData(data_object, Clipboard::GetUrlWFormatType()) ||
```

## bigvul / deadcode / bigvul_test_010144

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -1,3 +1,4 @@
 bool ValidateExperiment(const Experiment& e) {
+  { int __dc_0001 = 0; if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
   switch (e.type) {
     case Experiment::SINGLE_VALUE:
@@ -14,4 +15,5 @@
       NOTREACHED();
   }
+  { int __dc_0002 = 0; if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
   return true;
 }
```

## bigvul / deadcode / bigvul_test_010615

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -1,6 +1,8 @@
 struct crypto_ahash *mcryptd_ahash_child(struct mcryptd_ahash *tfm)
 {
+	{ int __dc_0001 = 0; if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
 	struct mcryptd_hash_ctx *ctx = crypto_ahash_ctx(&tfm->base);
 
+	{ int __dc_0002 = 0; if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
 	return ctx->child;
 }
```

## bigvul / deadcode / bigvul_test_010855

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -2,16 +2,21 @@
      const gs_gstate * pgs, gx_device * dev, gs_color_select_t select)
 {
+    { int __dc_0001 = 0; if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
     int i,ncomps;
     frac cm_comps[GX_DEVICE_COLOR_MAX_COMPONENTS];
+    { int __dc_0002 = 0; if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
     gx_color_value cv[GX_DEVICE_COLOR_MAX_COMPONENTS];
     gx_color_index color;
+    { int __dm_0003 = 0; for (int __fl_0004 = 0; __fl_0004 < 0; __fl_0004++) { __dm_0003 += __fl_0004; } }
     gx_device *trans_device;
 
     /* If trans device is set, we need to use its procs. */
     if (pgs->trans_device != NULL){
+        { int __dc_0005 = 0; if (__dc_0005 != 0) { __dc_0005 = __dc_0005 + 1; } }
         trans_device = pgs->trans_device;
     } else {
         trans_device = dev;
     }
+    { int __dc_0006 = 0; if (__dc_0006 != 0) { __dc_0006 = __dc_0006 + 1; } }
     ncomps = trans_device->color_info.num_components;
     /* map to the color model */
@@ -20,7 +25,9 @@
     /* If we are in an RGB blending color space and have spots then we have
     * possibly an issue here with the transfer function */
+    { int __dm_0007 = 0; for (int __fl_0008 = 0; __fl_0008 < 0; __fl_0008++) { __dm_0007 += __fl_0008; } }
     if (pgs->trans_device != NULL) {
         for (i = 0; i < 3; i++)
             cv[i] = frac2cv(gx_map_color_frac(pgs, cm_comps[i], effective_transfer[i]));
+        { int __dc_0009 = 0; if (__dc_0009 != 0) { __dc_0009 = __dc_0009 + 1; } }
         for (i = 3; i < ncomps; i++)
             cv[i] = gx_color_value_from_byte(cm_comps[i]);
```

## bigvul / deadcode / bigvul_test_011648

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -1,5 +1,7 @@
 static int on_body_fill_buffer(http_parser *parser, const char *str, size_t len)
 {
+	{ int __dc_0001 = (len ^ len); if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
 	parser_context *ctx = (parser_context *) parser->data;
+	{ int __dc_0002 = (len ^ len); if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
 	http_subtransport *t = ctx->t;
 
@@ -8,12 +10,17 @@
 	 * content anyway.
 	 */
+	{ int __dm_0003 = 0; for (int __fl_0004 = 0; __fl_0004 < 0; __fl_0004++) { __dm_0003 += __fl_0004; } }
 	if (t->parse_error == PARSE_ERROR_REPLAY)
 		return 0;
 
+	{ int __dc_0005 = (len ^ len); if (__dc_0005 != 0) { __dc_0005 = __dc_0005 + 1; } }
 	if (ctx->buf_size < len) {
+		{ int __dc_0006 = (len ^ len); if (__dc_0006 != 0) { __dc_0006 = __dc_0006 + 1; } }
 		giterr_set(GITERR_NET, "Can't fit data in the buffer");
+		{ int __dm_0007 = 0; for (int __fl_0008 = 0; __fl_0008 < 0; __fl_0008++) { __dm_0007 += __fl_0008; } }
 		return t->parse_error = PARSE_ERROR_GENERIC;
 	}
 
+	{ int __dc_0009 = (len ^ len); if (__dc_0009 != 0) { __dc_0009 = __dc_0009 + 1; } }
 	memcpy(ctx->buffer, str, len);
 	*(ctx->bytes_read) += len;
```

## bigvul / deadcode / bigvul_test_017506

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -1,4 +1,6 @@
    ~ScopedRequest() {
+    { int __dc_0001 = 0; if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
     if (requested_) {
+       { int __dc_0002 = 0; if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
        owner_->delegate_->StopEnumerateDevices(request_id_);
      }
```

## bigvul / deadcode / bigvul_test_018170

- [x] semantics preserved

```diff
--- original
+++ deadcode
@@ -1,3 +1,4 @@
 void OMXCodec::dumpPortStatus(OMX_U32 portIndex) {
+    { int __dc_0001 = 0; if (__dc_0001 != 0) { __dc_0001 = __dc_0001 + 1; } }
     OMX_PARAM_PORTDEFINITIONTYPE def;
  InitOMXParams(&def);
@@ -14,4 +15,5 @@
 
     printf("  nBufferCountActual = %" PRIu32 "\n", def.nBufferCountActual);
+    { int __dc_0002 = 0; if (__dc_0002 != 0) { __dc_0002 = __dc_0002 + 1; } }
     printf("  nBufferCountMin = %" PRIu32 "\n", def.nBufferCountMin);
     printf("  nBufferSize = %" PRIu32 "\n", def.nBufferSize);
@@ -26,4 +28,5 @@
             printf("  nFrameWidth = %" PRIu32 "\n", imageDef->nFrameWidth);
             printf("  nFrameHeight = %" PRIu32 "\n", imageDef->nFrameHeight);
+            { int __dm_0003 = 0; for (int __fl_0004 = 0; __fl_0004 < 0; __fl_0004++) { __dm_0003 += __fl_0004; } }
             printf("  nStride = %" PRIu32 "\n", imageDef->nStride);
 
@@ -44,4 +47,5 @@
             printf("  // Video\n");
             printf("  nFrameWidth = %" PRIu32 "\n", videoDef->nFrameWidth);
+            { int __dc_0005 = 0; if (__dc_0005 != 0) { __dc_0005 = __dc_0005 + 1; } }
             printf("  nFrameHeight = %" PRIu32 "\n", videoDef->nFrameHeight);
             printf("  nStride = %" PRIu32 "\n", videoDef->nStride);
@@ -62,4 +66,5 @@
             printf("\n");
             printf("  // Audio\n");
+            { int __dc_0006 = 0; if (__dc_0006 != 0) { __dc_0006 = __dc_0006 + 1; } }
             printf("  eEncoding = %s\n",
                    asString(audioDef->eEncoding));
@@ -75,4 +80,5 @@
 
                 printf("  nSamplingRate = %" PRIu32 "\n", params.nSamplingRate);
+                { int __dm_0007 = 0; for (int __fl_0008 = 0; __fl_0008 < 0; __fl_0008++) { __dm_0007 += __fl_0008; } }
                 printf("  nChannels = %" PRIu32 "\n", params.nChannels);
                 printf("  bInterleaved = %d\n", params.bInterleaved);
@@ -89,4 +95,5 @@
                 amr.nPortIndex = portIndex;
 
+                { int __dc_0009 = 0; if (__dc_0009 != 0) { __dc_0009 = __dc_0009 + 1; } }
                 err = mOMX->getParameter(
                         mNode, OMX_IndexParamAudioAmr, &amr, sizeof(amr));
```

## bigvul / controlflow / bigvul_test_000859

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -16,14 +16,16 @@
     }
 
-    tbl->tables[tbl->ntable].ptr = tbl1;
-    tbl->tables[tbl->ntable].col = col;
-    tbl->tables[tbl->ntable].row = row;
-    tbl->tables[tbl->ntable].indent = tbl->indent;
+    { int __st_0002 = 0;
+    while (__st_0002 < 4) {
+        if (__st_0002 == 0) { tbl->tables[tbl->ntable].ptr = tbl1; __st_0002 = 1; }
+        else if (__st_0002 == 1) { tbl->tables[tbl->ntable].col = col; __st_0002 = 2; }
+        else if (__st_0002 == 2) { tbl->tables[tbl->ntable].row = row; __st_0002 = 3; }
+        else if (__st_0002 == 3) { tbl->tables[tbl->ntable].indent = tbl->indent; __st_0002 = 4; }
+    } }
     tbl->tables[tbl->ntable].buf = newTextLineList();
     check_row(tbl, row);
-    if (col + 1 <= tbl->maxcol && tbl->tabattr[row][col + 1] & HTT_X)
-	tbl->tables[tbl->ntable].cell = tbl->cell.icell;
-    else
-	tbl->tables[tbl->ntable].cell = -1;
+    { int __p_0001 = (col + 1 <= tbl->maxcol && tbl->tabattr[row][col + 1] & HTT_X) ? 1 : 0;
+    if (__p_0001) { tbl->tables[tbl->ntable].cell = tbl->cell.icell; }
+    if (!__p_0001) { tbl->tables[tbl->ntable].cell = -1; } }
     tbl->ntable++;
 }
```

## bigvul / controlflow / bigvul_test_001658

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -1,4 +1,5 @@
 void disable_write_same(struct mapped_device *md)
 {
+	{ int __p_0001 = (0 == 0); if (__p_0001 && (__p_0001 == 1)) { __p_0001 = __p_0001; } }
 	struct queue_limits *limits = dm_get_queue_limits(md);
 
```

## bigvul / controlflow / bigvul_test_003468

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -24,7 +24,10 @@
   if (length < 16)
     return((StringInfo *) NULL);
-  profile=BlobToStringInfo((const unsigned char *) NULL,length);
-  SetStringInfoDatum(profile,blocks);
-  SetStringInfoName(profile,"8bim");
+  { int __st_0001 = 0;
+  while (__st_0001 < 3) {
+      if (__st_0001 == 0) { profile=BlobToStringInfo((const unsigned char *) NULL,length); __st_0001 = 1; }
+      else if (__st_0001 == 1) { SetStringInfoDatum(profile,blocks); __st_0001 = 2; }
+      else if (__st_0001 == 2) { SetStringInfoName(profile,"8bim"); __st_0001 = 3; }
+  } }
   for (p=blocks; (p >= blocks) && (p < (blocks+length-7)); )
   {
```

## bigvul / controlflow / bigvul_test_004376

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -3,4 +3,5 @@
         bool force_immediate_load,
         scoped_refptr<base::SequencedTaskRunner> background_task_runner) {
+  { int __p_0001 = (0 == 0); if (__p_0001 && (__p_0001 == 1)) { __p_0001 = __p_0001; } }
   return GetInstance()->CreateManagerForProfile(
       profile, force_immediate_load, background_task_runner);
```

## bigvul / controlflow / bigvul_test_004827

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -1,5 +1,5 @@
 void RenderViewImpl::OnFindReplyAck() {
   if (queued_find_reply_message_.get()) {
-    Send(queued_find_reply_message_.release());
+    { if ((1)) { Send(queued_find_reply_message_.release()); } }
   }
 }
```

## bigvul / controlflow / bigvul_test_007614

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -7,4 +7,4 @@
 
   if (View())
-    DispatchPrintEventRecursively(EventTypeNames::afterprint);
+    { if ((1)) { DispatchPrintEventRecursively(EventTypeNames::afterprint); } }
 }
```

## bigvul / controlflow / bigvul_test_014570

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -1,7 +1,7 @@
   void AddServiceRequestHandler(const std::string& name,
                                 const ServiceRequestHandler& handler) {
-    io_task_runner_->PostTask(
+    { if ((1)) { io_task_runner_->PostTask(
         FROM_HERE, base::Bind(&ServiceManagerConnectionImpl::IOThreadContext::
                                   AddServiceRequestHandlerOnIoThread,
-                              this, name, handler));
+                              this, name, handler)); } }
   }
```

## bigvul / controlflow / bigvul_test_015963

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -1,3 +1,4 @@
   bool has_web_ui_bindings() const {
+    { int __p_0001 = (0 == 0); if (__p_0001 && (__p_0001 == 1)) { __p_0001 = __p_0001; } }
     return enabled_bindings_ & BINDINGS_POLICY_WEB_UI;
   }
```

## bigvul / controlflow / bigvul_test_017585

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -4,17 +4,22 @@
 	struct hw_perf_event *hwc = &event->hw;
 
-	hwc->idx = cpuc->assign[i];
-	hwc->last_cpu = smp_processor_id();
-	hwc->last_tag = ++cpuc->tags[i];
+	{ int __st_0002 = 0;
+	while (__st_0002 < 3) {
+	    if (__st_0002 == 0) { hwc->idx = cpuc->assign[i]; __st_0002 = 1; }
+	    else if (__st_0002 == 1) { hwc->last_cpu = smp_processor_id(); __st_0002 = 2; }
+	    else if (__st_0002 == 2) { hwc->last_tag = ++cpuc->tags[i]; __st_0002 = 3; }
+	} }
 
 	if (hwc->idx == X86_PMC_IDX_FIXED_BTS) {
 		hwc->config_base = 0;
  		hwc->event_base	= 0;
- 	} else if (hwc->idx >= X86_PMC_IDX_FIXED) {
+ 	} else { int __p_0001 = (hwc->idx >= X86_PMC_IDX_FIXED) ? 1 : 0;
+    if (__p_0001) {
  		hwc->config_base = MSR_ARCH_PERFMON_FIXED_CTR_CTRL;
 		hwc->event_base = MSR_ARCH_PERFMON_FIXED_CTR0;
- 	} else {
+ 	}
+    if (!__p_0001) {
  		hwc->config_base = x86_pmu_config_addr(hwc->idx);
  		hwc->event_base  = x86_pmu_event_addr(hwc->idx);
-	}
+	} }
 }
```

## bigvul / controlflow / bigvul_test_017763

- [x] semantics preserved

```diff
--- original
+++ controlflow
@@ -1,4 +1,4 @@
 void GDataCache::Destroy() {
-  AssertOnSequencedWorkerPool();
+  { if ((1)) { AssertOnSequencedWorkerPool(); } }
   delete this;
 }
```
