package com.example.mobilefoodfreshness.ircamera

import android.content.Context
import android.content.Intent
import android.net.ConnectivityManager
import android.net.LinkAddress
import android.net.NetworkCapabilities
import android.net.nsd.NsdManager
import android.net.nsd.NsdServiceInfo
import android.net.wifi.WifiManager
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import com.example.mobilefoodfreshness.R
import kotlinx.coroutines.*
import java.net.Inet4Address
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.Socket
import kotlin.math.min

class RobotPickerActivity : AppCompatActivity() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)

    private lateinit var listView: ListView
    private lateinit var progress: ProgressBar
    private lateinit var emptyTv: TextView
    private val foundIps = mutableListOf<String>()
    private lateinit var adapter: ArrayAdapter<String>

    private val nameByIp = mutableMapOf<String, String>()
    private var fallbackCounter = 0

    private var nsdManager: NsdManager? = null
    private var discoveryListener: NsdManager.DiscoveryListener? = null
    private var multicastLock: WifiManager.MulticastLock? = null

    private val targetPort = 9001
    private val chunkParallel = 64
    private val connectTimeoutMs = 600

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_device_picker)

        listView = findViewById(R.id.listView)
        progress = findViewById(R.id.progress)

        emptyTv = TextView(this).apply {
            text = "Device Not Found"
            visibility = View.GONE
            setPadding(24, 24, 24, 24)
        }
        (listView.parent as? LinearLayout)?.addView(emptyTv)

        adapter = ArrayAdapter(this, android.R.layout.simple_list_item_1, mutableListOf())
        listView.adapter = adapter

        listView.setOnItemClickListener { _, _, pos, _ ->
            val ip = foundIps[pos]
            startActivity(Intent(this, RosCameraActivity::class.java).apply {
                putExtra("robot_ip", ip)
                putExtra("robot_port", targetPort)
                putExtra("camera_topic", "/bot5/camera_node/image/compressed")
            })
        }
    }

    override fun onStart() {
        super.onStart()
        startNsd()
        scanSubnet()
    }

    override fun onStop() {
        super.onStop()
        stopNsd()
        scope.coroutineContext.cancelChildren()
    }


    private fun scanSubnet() {
        progress.visibility = View.VISIBLE
        emptyTv.visibility = View.GONE
        foundIps.clear()
        nameByIp.clear()
        fallbackCounter = 0
        adapter.clear()

        val (localIp, prefix) = getLocalIpv4AndPrefix() ?: run {
            progress.visibility = View.GONE
            emptyTv.visibility = View.VISIBLE
            Toast.makeText(this, "not connect to WIFI", Toast.LENGTH_LONG).show()
            return
        }

        val hosts = enumerateHosts(localIp, prefix)
        Log.i("RobotPicker", "local=$localIp/$prefix, scanHosts=${hosts.size}")

        scope.launch {
            withContext(Dispatchers.IO) {
                hosts
                    .filter { it != localIp }
                    .chunked(chunkParallel)
                    .forEach { chunk ->
                        coroutineScope {
                            chunk.map { ip ->
                                async {
                                    try {
                                        Socket().use { s ->
                                            s.connect(InetSocketAddress(ip, targetPort), connectTimeoutMs)
                                            ip
                                        }
                                    } catch (_: Exception) { null }
                                }
                            }.awaitAll().filterNotNull().forEach { okIp ->
                                withContext(Dispatchers.Main) {
                                    if (!foundIps.contains(okIp)) {
                                        foundIps.add(okIp)
                                        if (!nameByIp.containsKey(okIp)) {
                                            fallbackCounter += 1
                                            nameByIp[okIp] = "Device$fallbackCounter"
                                        }
                                        refreshListUI()
                                    }
                                }
                            }
                        }
                    }
            }
            progress.visibility = View.GONE
            emptyTv.visibility = if (foundIps.isEmpty()) View.VISIBLE else View.GONE
            if (foundIps.isEmpty()) {
                Toast.makeText(
                    this@RobotPickerActivity,
                    "not found rosbridge($targetPort).",
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }

    private fun getLocalIpv4AndPrefix(): Pair<String, Int>? {
        val cm = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val net = cm.activeNetwork ?: return null
        val caps = cm.getNetworkCapabilities(net) ?: return null
        if (!caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI)) return null
        val lp = cm.getLinkProperties(net) ?: return null

        val v4: LinkAddress = lp.linkAddresses.firstOrNull { it.address is Inet4Address } ?: return null
        val ip = v4.address.hostAddress ?: return null
        return ip to v4.prefixLength
    }

    private fun enumerateHosts(ip: String, prefix: Int): List<String> {
        val addr = InetAddress.getByName(ip).address // big-endian bytes
        val ipInt = ((addr[0].toInt() and 0xFF) shl 24) or
                ((addr[1].toInt() and 0xFF) shl 16) or
                ((addr[2].toInt() and 0xFF) shl 8) or
                (addr[3].toInt() and 0xFF)
        val mask = if (prefix <= 0) 0 else (-1 shl (32 - prefix))
        val network = ipInt and mask
        val broadcast = network or mask.inv()

        val total = (broadcast.toLong() - network.toLong() - 1).toInt().coerceAtLeast(0)
        val maxScan = min(total, 4096)
        val list = ArrayList<String>(maxScan)

        var cur = network + 1
        var count = 0
        while (cur < broadcast && count < maxScan) {
            list.add(intToIp(cur))
            cur++; count++
        }
        return list
    }

    private fun intToIp(v: Int): String {
        val b0 = (v ushr 24) and 0xFF
        val b1 = (v ushr 16) and 0xFF
        val b2 = (v ushr 8) and 0xFF
        val b3 = v and 0xFF
        return "$b0.$b1.$b2.$b3"
    }


    private fun refreshListUI() {
        val items = foundIps
            .sortedWith(compareByDescending<String> { nameByIp.containsKey(it) }.thenBy { it })
            .map { ip ->
                val name = nameByIp[ip] ?: "Device?"
                "✅ $name ($ip:$targetPort)"
            }
        adapter.clear()
        adapter.addAll(items)
        adapter.notifyDataSetChanged()
    }


    private fun startNsd() {
        runCatching {
            val wm = applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
            multicastLock = wm.createMulticastLock("mdns").apply { setReferenceCounted(true); acquire() }
        }

        nsdManager = getSystemService(Context.NSD_SERVICE) as NsdManager
        discoveryListener = object : NsdManager.DiscoveryListener {
            override fun onDiscoveryStarted(serviceType: String) {}
            override fun onDiscoveryStopped(serviceType: String) {}
            override fun onStartDiscoveryFailed(serviceType: String, errorCode: Int) { stopNsd() }
            override fun onStopDiscoveryFailed(serviceType: String, errorCode: Int) { stopNsd() }
            override fun onServiceLost(serviceInfo: NsdServiceInfo) {}

            override fun onServiceFound(serviceInfo: NsdServiceInfo) {
                val t = serviceInfo.serviceType?.lowercase() ?: return
                if (!t.contains("_rosbridge._tcp")) return

                nsdManager?.resolveService(serviceInfo, object : NsdManager.ResolveListener {
                    override fun onResolveFailed(info: NsdServiceInfo, errorCode: Int) {}
                    override fun onServiceResolved(info: NsdServiceInfo) {
                        val host = info.host?.hostAddress ?: return
                        val port = info.port
                        val name = info.serviceName ?: "rosbridge"
                        if (port == targetPort) {
                            nameByIp[host] = name
                            if (foundIps.contains(host)) refreshListUI()
                        }
                    }
                })
            }
        }
        nsdManager?.discoverServices("_rosbridge._tcp.", NsdManager.PROTOCOL_DNS_SD, discoveryListener)
    }

    private fun stopNsd() {
        runCatching { discoveryListener?.let { nsdManager?.stopServiceDiscovery(it) } }
        discoveryListener = null
        nsdManager = null
        runCatching { multicastLock?.let { if (it.isHeld) it.release() } }
        multicastLock = null
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel()
    }
}
